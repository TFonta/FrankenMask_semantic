python train_frankenmask.py --name RGB_model_no_bg --load_size 256 \
    --sample_dir samples/512/ \
    --checkpoints_dir checkpoints/512/ \
    --batchSize 64 --crop_size 256 --style_dim 512 --dataset_mode custom --label_dir datasets/CelebA-HQ/train/labels \
    --image_dir datasets/CelebA-HQ/train/images --label_dir_test datasets/CelebA-HQ/test/labels \
    --image_dir_test /datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0