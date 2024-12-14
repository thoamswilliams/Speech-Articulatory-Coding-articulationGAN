from sparc import load_model
import argparse
import tqdm
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
#parser.add_argument("--rank",type=int,default=0)
#parser.add_argument("--n",type=int,default=1)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--wav_dir", type=str, )
parser.add_argument("--save_dir", type=str, )
#parser.add_argument("--batch_size",type=int,default=1)

if __name__ == "__main__":
    
    args = parser.parse_args()
    device = args.device
    wav_dir = Path(args.wav_dir)
    save_dir = Path(args.save_dir)
    spk_emb_save_dir = save_dir/"spk_emb"
    spk_emb_save_dir.mkdir(exist_ok=True)
    ft_save_dir = save_dir/"emasrc"
    ft_save_dir.mkdir(exist_ok=True)
    
    config_path = "../configs/feature_extraction.yaml"
    coder = load_model("feature_extraction",
                       config=config_path, device=device) 
    
    wav_files = [f for f in wav_dir.glob("**/*.flac")] + [f for f in wav_dir.glob("**/*.wav")]
    
    for wav_file in tqdm.tqdm(wav_files):
        outputs = coder.encode(wav_file, concat=True)
        
        save_name = str(wav_file).replace(str(wav_dir),"")
        save_name = Path(save_name).stem+".npy"
        
        def _recursive_path_solver(file_path):
            if file_path.exists():
                return
            elif file_path.parent.exists():
                file_path.mkdir(exist_ok=True)
                return
            else:
                _recursive_path_solver(file_path.parent)
                
        path_depth = len(save_name.split("/"))
        
        ft_save_path = ft_save_dir/save_name
        spk_emb_save_path = spk_emb_save_dir/save_name
        
        _recursive_path_solver(spk_emb_save_path.parent)
        _recursive_path_solver(ft_save_path.parent)
        
        np.save(ft_save_path, outputs["features"])
        np.save(spk_emb_save_path, outputs["spk_emb"])
        
            
            
        
        
        