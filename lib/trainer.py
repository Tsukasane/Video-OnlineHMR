import torch
import logging
from tqdm import tqdm
from lib.core.base_trainer import BaseTrainer
from lib.utils.pose_utils import Evaluator

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):

    def _init_fn(self):
        return

    def train_one_epoch(self, ):
        ##### NOTE(yiwen) the train function
        self.model.train()
        self.model.freeze_modules()
        update_iter = self.cfg.TRAIN.UPDATE_ITER
        crop_size = self.model.crop_size

        self.valid_range = self.cfg.MODEL.VALID_RANGE # prev 0, curr 1, future 2

        for i, batch in enumerate(tqdm(self.train_loader, desc="Computing batch")): # how to ignore the train.invalid elements

            # 72, 24, 4     B*window_size, 24, 4'
            
            batch = {k: v.flatten(0, 1) for k, v in batch.items() if type(v)==torch.Tensor}
            # ['img_idx', 'img_focal', 'img_center', 'img', 'pose', 'betas', 'pose_3d', 'keypoints', 'scale', 'center', 'has_smpl', 'has_pose_3d']
            # [26, 1]  [26]  [26, 2]  [26, 3, 256, 256]  [26, 72]  [26, 10]  [26, 24, 4]  [26, 49, 3]  [26]  [26, 2]  [26]  [26]

            batch = {k: v.to(self.device) for k, v in batch.items()} # continuous

            if batch['img'].shape[0] < self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQ_LEN:
                continue
            
            batch['beta_weight'] = self.cfg.TRAIN.SMPL_BETA
            batch['smpl'] = self.model.smpl

            # Forward pass
            out, iter_preds = self.model(batch, self.valid_range, is_train=True, iters=update_iter)
            try:
                batch['pred_rotmat_0'] = out['pred_rotmat_0'] # 72, 24, 3, 3
            except Exception:
                batch['pred_rotmat_0'] = None

            # Loss on full sequence
            rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds = iter_preds
            N = len(rotmat_preds)
            
            gamma = self.cfg.TRAIN.GAMMA
            train_bs = self.cfg.TRAIN.BATCH_SIZE
            loss = 0

            for j in range(N):
                batch['pred_rotmat'] = rotmat_preds[j] # B*T=72, 24, 3, 3
                batch['pred_betas'] = shape_preds[j] # 72, 10
                batch['pred_cam'] = cam_preds[j] # 72, 3
                batch['pred_keypoints_3d'] = j3d_preds[j] # 72, 49, 3
                batch['pred_keypoints_2d'] = (j2d_preds[j]-crop_size/2.) / (crop_size/2.) # 72, 49, 2
                
                loss_j, losses = self.criterion(batch, self.valid_range, train_bs)
                loss += gamma**(N-j-1) * loss_j
                
            loss *= self.cfg.TRAIN.LOSS_SCALE

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.cfg.TRAIN.CLIP_GRADIENT == True:
                self.clip_gradient_norm(self.model, max_norm=self.cfg.TRAIN.CLIP_NORM)

            self.optimizer.step()
            
            self.global_step += 1
            self.loss_meter.update(losses)
            self.lr_scheduler.step()

            self.check_and_validate(i)

            if self.should_break():
                break

        return 
        

    def check_and_validate(self, batch_id):
        ##### NOTE(yiwen) validation function
        steps = self.global_step

        # Training summary
        if steps % self.cfg.TRAIN.SUMMARY_STEP == 0:
            self.upload_losses(step = steps)
            self.upload_additional(step = steps)
            self.writer.flush()

        # Validation summary
        if steps % self.cfg.TRAIN.VALID_STEP == 0:
            if steps < 5000:
                save_best = False
            else:
                save_best = True
            
            performance = self.validate()
            self.check_performance(performance, batch=batch_id, save_best=save_best)
        else:
            performance = None

        
        # Checkpoint
        if steps % self.cfg.TRAIN.SAVE_STEP == 0:
            self.save_checkpoint(batch=batch_id, performance=performance,
                                index='_{:04d}'.format(steps))
            

    def validate(self,):
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validating ...")
        torch.cuda.empty_cache() 

        self.model.eval()
        update_iter = self.cfg.TRAIN.UPDATE_ITER

        model = self.model
        loader = self.test_loader
        device = self.device
        db = loader.dataset
        
        self.valid_range = self.cfg.MODEL.VALID_RANGE
       
        gt_vertices = None
        pred_vertices = None

        # evaluator = Evaluator(dataset_length=len(db.imgname),
        #                       seq_len=getattr(model, 'seq_len', None))
        evaluator = Evaluator(dataset_length=len(db.imgname),
                              seq_len=getattr(model, 'seq_len', None))
        J_regressor = db.J_regressor.to(device)

        for i, batch in enumerate(loader):
            # batch = loader.batch_normalize_img(batch)
            # NOTE(yiwen) no need to modify the batch in inference
            batch = {k: v.to(self.device).flatten(0, 1) for k, v in batch.items() if type(v)==torch.Tensor}
            # gt joints
            gt_keypoints_3d = batch['pose_3d'] # [bt, 24, 4] # TODO(yiwen) seperate video in validation

            # prediction
            with torch.no_grad():
                # batch.keys() ['img_idx', 'img_focal', 'img_center', 'img', 'pose', 'betas', 'pose_3d', 'gt_verts', 'keypoints', 'scale', 'center', 'has_smpl', 'has_pose_3d']

                # NOTE(yiwen) Option2: use the test/inference workflow
                out, _ = model.inference_forward(batch, is_valid=True) # default is inference mode
                
                # out.keys() 'pred_cam', 'pred_pose', 'pred_shape', 'pred_rotmat', 'pred_rotmat_0', 'trans_full'
                
                if '3dpw' in db.dataset: # TODO(yiwen) temporally use 3dpw as evalset to see vertices performance
                    mode = '3dpw'
                    smpl_out = model.smpl.query(out) # input ['pred_rotmat'] ['pred_shape']
                    pred_vertices = smpl_out.vertices # 48, 6890, 3
                    gt_vertices = batch['gt_verts']
                    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

                    pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                    pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
                    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis # NOTE(yiwen) move to pelvis (0,0)

                elif 'emdb' in db.dataset: # emdb_1 v
                    mode = 'emdb'
                    smpl_out = model.smpl.query(out, default_smpl=True)
                    pred_keypoints_3d = smpl_out.joints[:, :24]

                    pred_pelvis = pred_keypoints_3d[:,[1,2],:].mean(dim=1, keepdim=True).clone()
                    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis # NOTE(yiwen) only focus on relative motion, not absolute position
                    
            # evaluation
            evaluator(gt_keypoints_3d, pred_keypoints_3d, mode, gt_vertices, pred_vertices)

        re = evaluator.re[:evaluator.counter].mean()
        mpjpe = evaluator.mpjpe[:evaluator.counter].mean()
        acc = evaluator.acc[:evaluator.counter].mean()
        jitter = evaluator.jitter[:evaluator.counter].mean()
        jitter_gt = evaluator.jitter_gt[:evaluator.counter].mean()


        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation re: {re}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation mpjpe: {mpjpe}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation accel: {acc}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation jitter: {jitter}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation jitter: {jitter_gt}")

        self.writer.add_scalar(f"Validation/RE", re, self.global_step)
        self.writer.add_scalar(f"Validation/MPJPE", mpjpe, self.global_step)
        self.writer.add_scalar(f"Validation/ACCEL", acc, self.global_step)
        self.writer.add_scalar(f"Validation/JITTER", jitter, self.global_step)
        self.writer.add_scalar(f"Validation/JITTER_GT", jitter_gt, self.global_step)
        self.writer.flush()

        self.model.train()
        self.model.freeze_modules()

        self.performance_type = 'min'

        torch.cuda.empty_cache()
        return re

    def upload_additional(self, step):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("Z/lr", lr, step)
        
        return


    

