checkpoint_config = dict(interval=1)

log_config = dict(
    interval=250,
)
save_image_config = dict(
    interval=250,
)
optimizer = dict(type='Adam', lr=0.0004)

loss = dict(type='MSELoss')

runner = dict(max_epochs=300)

checkpoints = None
resume = None
