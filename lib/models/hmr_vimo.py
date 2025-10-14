import numpy as np
import einops
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from lib.utils.geometry import perspective_projection
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
from lib.datasets.track_dataset import TrackDataset

from .vit import vit_huge
from .modules import *
from .smpl import SMPL
from ..pipeline.tools import parse_chunks

from lib.models.casual_kvcache import SMPLDecoderModel

autocast = torch.amp.autocast

def select_valid(batch_tensor, batch_size):
    batch_tensor = batch_tensor.reshape(batch_size, -1, *batch_tensor.shape[1:])[:,2:,...]
    batch_tensor = batch_tensor.reshape(-1, *batch_tensor.shape[2:])

    return batch_tensor

class HMR_VIMO(nn.Module):
    def __init__(self, cfg=None, device='cuda', **kwargs):

        super(HMR_VIMO, self).__init__()
        self.device = device
        self.cfg = cfg
        self.crop_size = cfg.IMG_RES
        self.seq_len = cfg.DATASET.SEQ_LEN
        self.chunk_size = 16
        self.train_bs = cfg.TRAIN.BATCH_SIZE
        self.valid_bs = cfg.TEST.BATCH_SIZE

        # SMPL
        self.smpl = SMPL()      

        # Backbone
        self.backbone = vit_huge()

        # cache config
        self.max_memt = 2 # frame
        self.H = 16
        self.W = 12
        self.smpl_decoder = SMPLDecoderModel(input_dim=1280, 
                                             hidden_dim=512, 
                                             device=device,
                                             max_cache=self.max_memt * self.H * self.W) # t*h*w

        self.register_buffer('initialized', torch.tensor(False))
        self.inference_memory = None


    def forward(self, batch, valid_range=(0,2), is_train=False, is_valid=False, **kwargs):
        '''
        Args:
            - batch (dict)
                - batch['img'] # B*T, 3, 256, 256
                - batch['center'] # B*T, 2
                - batch['scale'] # B*T
                - batch['img_focal'] # B*T
                - batch['img_center'] # B*T, 2
            - valid_range: is for prev, curr, future ablation
        Returns:
            - out (dict)
                - rotmat_preds (list) element shape B*T, 24, 3, 3
                - shape_preds (list) element shape B*T, 10
                - cam_preds (list) element shape B*T, 3
                - j3d_preds (list) element shape B*T, 49, 3
                - j2d_preds (list) element shape B*T, 49, 2
                - trans_full (list) element shape B*T, 1, 3
        '''

        image = batch['img'] # B*T, 3, 256, 256
        center = batch['center'] # B*T, 2
        scale  = batch['scale'] # B*T
        img_focal = batch['img_focal'] # B*T
        img_center = batch['img_center'] # B*T, 2

        batch_size = self.train_bs

        # estimate focal length, and bbox 
        bbox_info = self.bbox_est(center, scale, img_focal, img_center) # 128, 3

        # backbone 
        with autocast('cuda'):
            # B*N*T=2*24*3, 3, H, W --> BT, C=1280, h, w
            feature = self.backbone(image[:,:,:,32:-32]) # pass through vit
            feature = feature.float() # 128, 1280, w=16, h=12 NOTE(yiwen) image feature of each patch/frame

        # frame level
        bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12) 
        # feature = torch.cat([feature, bb], dim=1) # B*3=48, 1283, 16, 12 NOTE(yiwen) image + human bbox --> if we don't use this bbox info

        # patch level -->
        feature = einops.rearrange(feature, '(b t) c h w -> b t (h w) c', b=batch_size) # c=1280 image feature only, use input in this shape to add spatial and temporal emcoding
    
        #### add new
        max_cache = self.max_memt * self.H * self.W
        total_num = feature.shape[1]
        chunks = feature.unfold(dimension=1, size=self.max_memt+1, step=1)
        chunks = chunks.reshape(-1,*chunks.shape[2:]).permute(0,3,1,2) # B*N, window_length, h*w, D
        q_token_raw = chunks[:,2:3,...] # current
        img_feats_raw = chunks[:,0:2,...] # previous

        new_bs = img_feats_raw.shape[0] # 336(24*(16-2)), 2, 192, 1280
        q_token2 = einops.rearrange(q_token_raw, 'b t (h w) c -> b (t h w) c', b=new_bs, h=self.H, w=self.W)

        pred_pose, pred_shape, pred_cam = self.smpl_decoder(img_feats_all=img_feats_raw, q_tokens=q_token2)
        pred_pose = pred_pose.reshape(-1, pred_pose.shape[-1]) # B*T, 144
        pred_shape = pred_shape.reshape(-1, pred_shape.shape[-1]) # B*T, 10
        pred_cam = pred_cam.reshape(-1, pred_cam.shape[-1])

        pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, 24, 3, 3)

        # Predictions
        rotmat_preds  = [] 
        shape_preds = []
        cam_preds   = []
        j3d_preds = []
        j2d_preds = []

        out = {}
        out['pred_cam'] = pred_cam # B*T, 3
        out['pred_pose'] = pred_pose # B*T, 144
        out['pred_shape'] = pred_shape # B*T, 10
        out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(-1, 24, 3, 3)
        out['pred_rotmat_0'] = pred_rotmat_0
        
        s_out = self.smpl.query(out)
        j3d = s_out.joints
        
        center = select_valid(center, batch_size)
        scale = select_valid(scale, batch_size)
        img_focal = select_valid(img_focal, batch_size)
        img_center = select_valid(img_center, batch_size)
        j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)

        rotmat_preds.append(out['pred_rotmat'].clone())
        shape_preds.append(out['pred_shape'].clone())
        cam_preds.append(out['pred_cam'].clone())
        j3d_preds.append(j3d.clone())
        j2d_preds.append(j2d.clone())
        iter_preds = [rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds]

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full
        
        return out, iter_preds
    
    def inference_forward(self, batch, valid_range=(0,2), is_train=False, is_valid=False, device='cuda', cache=None, **kwargs): # emdb2 eval
        '''
        TODO(yiwen) cache 需要传到这个函数外边，self.backbone 每次只提一个image的feature
        T=1
        Args:
            - batch (dict)
                - batch['img'] # B*T, 3, 256, 256
                - batch['center'] # B*T, 2
                - batch['scale'] # B*T
                - batch['img_focal'] # B*T
                - batch['img_center'] # B*T, 2
            - valid_range: is for prev, curr, future ablation
        Returns:
            - out (dict)
                - rotmat_preds (list) element shape B*T, 24, 3, 3
                - shape_preds (list) element shape B*T, 10
                - cam_preds (list) element shape B*T, 3
                - j3d_preds (list) element shape B*T, 49, 3
                - j2d_preds (list) element shape B*T, 49, 2
                - trans_full (list) element shape B*T, 1, 3
        '''

        image = batch['img'] # B*T, 3, 256, 256
        center = batch['center'] # B*T, 2
        scale  = batch['scale'] # B*T
        img_focal = batch['img_focal'] # B*T
        img_center = batch['img_center'] # B*T, 2

        if is_train:
            batch_size = self.train_bs # TODO(yiwen) pass through configs to function
        if is_valid:
            batch_size = self.valid_bs
        else:
            batch_size = 1

        # estimate focal length, and bbox 
        bbox_info = self.bbox_est(center, scale, img_focal, img_center) # 128, 3

        # backbone 
        with autocast('cuda'):
            # B*N*T=2*24*3, 3, H, W --> BT, C=1280, h, w
            feature = self.backbone(image[:,:,:,32:-32]) # pass through vit
            feature = feature.float() # 128, 1280, w=16, h=12 NOTE(yiwen) image feature of each patch/frame

        # frame level
        bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12) 
        # feature = torch.cat([feature, bb], dim=1) # B*3=48, 1283, 16, 12 NOTE(yiwen) image + human bbox --> if we don't use this bbox info

        # patch level -->
        feature = einops.rearrange(feature, '(b t) c h w -> b t (h w) c', b=batch_size) # c=1280 image feature only
        debug = True
        if not is_train: # in inference / validation
            inference_seqlen = feature.shape[1]

            pred_pose, pred_shape, pred_cam = [], [], []
            for t in range(inference_seqlen):
                
                q_token_raw = feature[:,t:t+1,...]
                new_bs = q_token_raw.shape[0] # 336(24*(16-2)), 2, 192, 1280
                q_token2 = einops.rearrange(q_token_raw, 'b t (h w) c -> b (t h w) c', b=new_bs, h=self.H, w=self.W)
                img_feats_raw = feature[:,t:t+1,...] # init
                smpl_pose, smpl_shape, smpl_cam, cache = self.smpl_decoder.inference_step(img_feat_t=img_feats_raw, 
                                                                                          q_tokens=q_token2, 
                                                                                          t=t, 
                                                                                          device=device, 
                                                                                          cache=cache)
    
                pred_pose.append(smpl_pose.unsqueeze(1))
                pred_shape.append(smpl_shape.unsqueeze(1))
                pred_cam.append(smpl_cam.unsqueeze(1))

            pred_pose = torch.cat(pred_pose, dim=1)
            pred_shape = torch.cat(pred_shape, dim=1)
            pred_cam = torch.cat(pred_cam, dim=1)
            
        else: # only for training pipeline debug

            #### add new
            print(f"debug -- valid using train pipeline")
            max_cache = self.max_memt * self.H * self.W
            total_num = feature.shape[1]
            chunks = feature.unfold(dimension=1, size=self.max_memt+1, step=1)
            chunks = chunks.reshape(-1,*chunks.shape[2:]).permute(0,3,1,2) # B*N, window_length, h*w, D
            q_token_raw = chunks[:,2:3,...] # current
            img_feats_raw = chunks[:,0:2,...] # previous

            new_bs = img_feats_raw.shape[0] # 336(24*(16-2)), 2, 192, 1280
            q_token2 = einops.rearrange(q_token_raw, 'b t (h w) c -> b (t h w) c', b=new_bs, h=self.H, w=self.W)
            # 16*(8-2)
            pred_pose, pred_shape, pred_cam = self.smpl_decoder(img_feats_all=img_feats_raw, q_tokens=q_token2)

        pred_pose = pred_pose.reshape(-1, pred_pose.shape[-1]) # B*T, 144
        pred_shape = pred_shape.reshape(-1, pred_shape.shape[-1]) # B*T, 10
        pred_cam = pred_cam.reshape(-1, pred_cam.shape[-1])

        pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, 24, 3, 3)

        # Predictions
        rotmat_preds  = [] 
        shape_preds = []
        cam_preds   = []
        j3d_preds = []
        j2d_preds = []

        # out = {}
        # out['pred_cam'] = pred_cam # B*T, 3
        # out['pred_pose'] = pred_pose # B*T, 144
        # out['pred_shape'] = pred_shape # B*T, 10
        # out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(-1, 24, 3, 3)
        # out['pred_rotmat_0'] = pred_rotmat_0

        # s_out = self.smpl.query(out)
        # j3d = s_out.joints
    
        out = {}
        out['pred_cam'] = select_valid(pred_cam, batch_size) # B*T, 3
        out['pred_pose'] = select_valid(pred_pose, batch_size) # B*T, 144
        out['pred_shape'] = select_valid(pred_shape, batch_size) # B*T, 10
        out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(-1, 24, 3, 3)
        out['pred_rotmat_0'] = select_valid(pred_rotmat_0, batch_size)
        
        s_out = self.smpl.query(out)
        j3d = s_out.joints

        if debug:
            # print(f"debug -- in valid")
            center = select_valid(center, batch_size)
            scale = select_valid(scale, batch_size)
            img_focal = select_valid(img_focal, batch_size)
            img_center = select_valid(img_center, batch_size)

        # if 16/16, then the first two frames in a seq may have relatively low performance due to the insufficient cache
        j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)
        rotmat_preds.append(out['pred_rotmat'].clone())
        shape_preds.append(out['pred_shape'].clone())
        cam_preds.append(out['pred_cam'].clone())
        j3d_preds.append(j3d.clone())
        j2d_preds.append(j2d.clone())
        iter_preds = [rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds]

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full

        return out, iter_preds, cache # TODO(yiwen) add return cache
    

    def inference_chunk_ar(self, imgfiles, boxes, img_focal, img_center, device='cuda', cache=None): # for vis
        db = TrackDataset(imgfiles, boxes, img_focal=img_focal, 
                        img_center=img_center, normalization=True, dilate=1.2)

        items = []
        for i in tqdm(range(len(db))):
            item = db[i] # dict
            items.append(item)

        batch = default_collate(items) # all-to-one-batch, but frame by frame forward process
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
    
            out, _, cache = self.inference_forward(batch, cache=cache)

        results = {'pred_cam': out['pred_cam'].cpu(),
                'pred_pose': out['pred_pose'].cpu(),
                'pred_shape': out['pred_shape'].cpu(),
                'pred_rotmat': out['pred_rotmat'].cpu(),
                'pred_trans': out['trans_full'].cpu(),
                'img_focal': img_focal,
                'img_center': img_center}
        
        return results, cache


    # def inference_ar_online(self, imgfiles, boxes, img_focal=None, img_center=None, valid=None, frame=None, device='cuda'):
    #     """
    #     imgfiles: (3,) numpy.array 3 frames chunk, each time inference the result of the last frame
    #     boxes: (3, 5) 3 frames boxes, the last dim is confidence
    #     """

    #     # TODO(yiwen) remove the boxes?

    #     # NOTE(yiwen) this chunk is only for all tracking results
    #     results = self.inference_chunk_ar(imgfiles, boxes, img_focal=img_focal, img_center=img_center)
        
    #     return results



    def inference_ar(self, imgfiles, boxes, img_focal=None, img_center=None, valid=None, frame=None, device='cuda'):
        nfile = len(imgfiles)
        if valid is None:
            valid = np.ones(nfile, dtype=bool)
        if frame is None:
            frame = np.arange(nfile)
        
        if isinstance(imgfiles, list):
            imgfiles = np.array(imgfiles)

        frame = frame[valid] # (129,)
        boxes = boxes[valid] # (129, 5)

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=3) # NOTE(yiwen) only segment if have missing tracking frames
        # boxes_chunks[0].shape (129, 5) frame_chunks[0].shape (129,)
        if len(frame_chunks) == 0:
            return

        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []
        frame = []

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles[frame_ck]
            # NOTE(yiwen) this chunk is only for all tracking results
            results = self.inference_chunk_ar(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center)

            pred_cam.append(results['pred_cam'])
            pred_pose.append(results['pred_pose'])
            pred_shape.append(results['pred_shape'])
            pred_rotmat.append(results['pred_rotmat'])
            pred_trans.append(results['pred_trans'])
            frame.append(torch.from_numpy(frame_ck))

        results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans),
                'frame': torch.cat(frame)}
        
        return results


    def inference(self, imgfiles, boxes, img_focal=None, img_center=None, valid=None, frame=None, device='cuda'):
        '''
        Args:
            - imgfiles (List): image paths
            - 
        '''
        nfile = len(imgfiles)
        if valid is None:
            valid = np.ones(nfile, dtype=bool)
        if frame is None:
            frame = np.arange(nfile)
        
        if isinstance(imgfiles, list):
            imgfiles = np.array(imgfiles)

        frame = frame[valid] # (129,)
        boxes = boxes[valid] # (129, 5)

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=3) # NOTE(yiwen) only segment if have missing tracking frames
        # boxes_chunks[0].shape (129, 5) frame_chunks[0].shape (129,)
        if len(frame_chunks) == 0:
            return

        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []
        frame = []

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles[frame_ck]
            results = self.inference_chunk(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center)

            pred_cam.append(results['pred_cam'])
            pred_pose.append(results['pred_pose'])
            pred_shape.append(results['pred_shape'])
            pred_rotmat.append(results['pred_rotmat'])
            pred_trans.append(results['pred_trans'])
            frame.append(torch.from_numpy(frame_ck))

        results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans),
                'frame': torch.cat(frame)}
        
        return results


    def inference_chunk(self, imgfiles, boxes, img_focal, img_center, device='cuda'):
        db = TrackDataset(imgfiles, boxes, img_focal=img_focal, 
                        img_center=img_center, normalization=True, dilate=1.2)

        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        # To-do: efficient implementation with batch
        items = []
        for i in tqdm(range(len(db))):

            item = db[i] # dict
            items.append(item)

            if len(items) < self.seq_len: # 攒到seq_len 的长度
                continue
            elif len(items) == self.seq_len:
                batch = default_collate(items)
            else: # len(items) > self.seq_len
                items.pop(0) # first in first out
                batch = default_collate(items) # sliding window step=1

            # each batch is a three-frames window
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
                # batch.keys() 'img', 'img_idx', 'scale', 'center', 'img_focal', 'img_center'
                out, _ = self.forward(batch) 
                # out.keys() 'pred_cam', 'pred_pose', 'pred_shape', 'pred_rotmat', 'pred_rotmat_0', 'trans_full'
                
                if out['pred_cam'].shape[0] == 3: # prev+curr+future
                # NOTE(yiwen) we only use the estimation of current frame
                    out = {k:v[1:-1] for k,v in out.items()}

                elif out['pred_cam'].shape[0] == 2: # prev+curr
                    out = {k:v[1:] for k,v in out.items()}

            pred_cam.append(out['pred_cam'].cpu())
            pred_pose.append(out['pred_pose'].cpu())
            pred_shape.append(out['pred_shape'].cpu())
            pred_rotmat.append(out['pred_rotmat'].cpu())
            pred_trans.append(out['trans_full'].cpu())

        results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans),
                'img_focal': img_focal,
                'img_center': img_center}
        
        return results


    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = perspective_projection(points, rotation=None, translation=None,
                        focal_length=img_focal, camera_center=img_center)

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:,None]/2)[:,None,:]
        points2d = points2d * (self.crop_size / b)[:,None,None]

        if return_full:
            return points2d_full, points2d
        else:
            return points2d


    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b      = scale * 200
        cx, cy = center[:,0], center[:,1]            # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:,0], img_center[:,1]  # center of original image
        
        bs = b*s
        tx_full = tx + 2*(cx-img_cx)/bs
        ty_full = ty + 2*(cy-img_cy)/bs
        tz_full = 2*img_focal/bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full


    def bbox_est(self, center, scale, img_focal, img_center):
        '''
        Pixel representation
        '''
        # Original image center
        img_cx, img_cy = img_center[:,0], img_center[:,1]

        # Implement CLIFF (Li et al.) bbox feature
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8 
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)  

        return bbox_info


    def set_smpl_mean(self, ):
        SMPL_MEAN_PARAMS = 'data/smpl/smpl_mean_params.npz'

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def freeze_modules(self):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = False
            else:
                module.eval()
                for p in module.parameters(): p.requires_grad=False

        return


    def unfreeze_modules(self, ):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = True
            else:
                module.train()
                for p in module.parameters(): p.requires_grad=True

        self.frozen_modules = None

        return

