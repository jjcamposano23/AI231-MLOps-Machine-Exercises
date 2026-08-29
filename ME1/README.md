# AI 231 (MLOps) — Machine Exercise 1

3-layer CNN for MNIST classification. Every layer/operation (convolution, max
pool, flatten, linear, ReLU, log-softmax, cross-entropy) is implemented **from
scratch** with torch tensors + `einops`/`einsum`. No CNN libraries are used
(`nn.Conv2d`, `F.conv2d`, `nn.MaxPool2d`, `nn.Linear` are all avoided).
`torch.autograd` handles backprop and `torch.optim.Adam` is the optimizer.

## Files
| File | Purpose |
|---|---|
| `me1_cnn_mnist.ipynb` | Main deliverable — notebook with results + 4×4 sample grid |
| `me1_cnn_mnist.py`    | Same logic as a runnable script |
| `requirements.txt`    | Dependencies |

## Architecture
```
28x28 --conv1(1->8,3x3)--relu--maxpool--> 8x13x13
      --conv2(8->16,3x3)-relu--maxpool--> 16x5x5
      --conv3(16->32,3x3)-relu---------->  32x3x3
      --flatten(288)--linear-->  10 logits
```
Trained 5 epochs, Adam(lr=1e-3), batch 128. Expected test accuracy ≈ 98–99%.

---

## Run on the SHARC DGX (A100)

**1. Copy this folder to the HPC** — from Windows PowerShell:
```powershell
scp -r "C:\Users\Jomar\Documents\Grad School\Coursework\26-27_1S_AI231\ME 1" jomar.christian.camposano@ai-n002.hpc.coe.upd.edu.ph:~/ME1
```

**2. SSH in and build the environment** (Python 3.11 — torch has no 3.14 wheels yet):
```bash
ssh jomar.christian.camposano@ai-n002.hpc.coe.upd.edu.ph
module load conda
conda create -y -p ~/me1-env python=3.11
conda activate ~/me1-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install einops matplotlib jupyter nbconvert
```

**3. Pin to 2 free GPUs and execute the notebook headless.**
GPUs 0–5 were idle in your `nvidia-smi`; the model is tiny so it uses one device
(`cuda:0`), but this reserves 2 per the exercise:
```bash
cd ~/ME1
export CUDA_VISIBLE_DEVICES=0,1
jupyter nbconvert --to notebook --execute --inplace me1_cnn_mnist.ipynb
```
This runs all cells and saves outputs (accuracy prints + the 4×4 grid image) back
into the `.ipynb`. To just check accuracy quickly instead: `python me1_cnn_mnist.py`.

**4. Copy the executed notebook back to Windows** — from PowerShell:
```powershell
scp jomar.christian.camposano@ai-n002.hpc.coe.upd.edu.ph:~/ME1/me1_cnn_mnist.ipynb "C:\Users\Jomar\Documents\Grad School\Coursework\26-27_1S_AI231\ME 1\me1_cnn_mnist.ipynb"
scp jomar.christian.camposano@ai-n002.hpc.coe.upd.edu.ph:~/ME1/sample_predictions.png "C:\Users\Jomar\Documents\Grad School\Coursework\26-27_1S_AI231\ME 1\sample_predictions.png"
```

---

## Upload to GitHub

From Windows PowerShell (repo: `AI231-MLOps-Machine-Exercises`):
```powershell
cd "C:\Users\Jomar\Documents\Grad School\Coursework\26-27_1S_AI231"
git clone https://github.com/jjcamposano23/AI231-MLOps-Machine-Exercises.git
Copy-Item -Recurse "ME 1" "AI231-MLOps-Machine-Exercises\ME1"
cd AI231-MLOps-Machine-Exercises
git add ME1
git commit -m "Add Machine Exercise 1: 3-layer scratch CNN (einops/einsum) for MNIST"
git push
```
If the repo already contains work, replace the clone step with a `git pull` inside
your existing local copy.
