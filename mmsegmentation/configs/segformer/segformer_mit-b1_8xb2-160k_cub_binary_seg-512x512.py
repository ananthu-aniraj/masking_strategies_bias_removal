_base_ = ['./segformer_mit-b0_8xb2-160k_cub_binary_seg-512x512.py']
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'
# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                     init_kwargs={'project': 'cub_binary_seg', 'entity': 'ananthu-phd', 'job_type': 'train',
                                  'group': 'segformer_mit-b1',
                                  })]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth"
train_dataloader = dict(batch_size=8, num_workers=2)