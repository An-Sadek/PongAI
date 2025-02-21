import os

def check_checkpoints():
    checkpoint_dir = "neat-checkpoint"
    
    if not os.path.exists(checkpoint_dir):
        print("Thư mục neat-checkpoint không tồn tại.")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]
    
    if checkpoint_files:
        checkpoint_files.sort()  
        print("Các tệp checkpoint hiện có:")
        for file in checkpoint_files:
            print(file)
    else:
        print("Không có tệp checkpoint nào.")

check_checkpoints()
