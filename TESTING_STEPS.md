# Testing Steps - Simple Instructions

Follow these steps **in order** to verify your APE-data works with the model.

## Quick Start (Automated)

```bash
# 1. Navigate to project directory
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v

# 2. Make sure you're on the pretrained_main branch (IMPORTANT!)
git checkout pretrained_main

# 3. Run the automated test script
./RUN_TESTS.sh
```

That's it! The script will:
- Check your Python version
- Install all dependencies
- Run comprehensive tests
- Tell you if everything works

**Time:** 5-10 minutes

---

## Manual Steps (If you prefer step-by-step)

### Step 1: Setup Environment (5 minutes)

```bash
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v
git checkout pretrained_main
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Tests (2 minutes)

```bash
python test_ape_data_loading.py
```

**Expected:** All 3 tests pass (✓ PASS)

---

## What Gets Tested?

1. ✅ **DICOM Loading** - Can read medical CT scans from zip files
2. ✅ **Data Format** - Correct tensor shapes (B, C, T, H, W)
3. ✅ **Model Integration** - Model processes APE data without errors

---

## If Tests Pass

You'll see:
```
🎉 All tests passed! Your setup is ready for training.
```

**Next steps:**
1. *(Optional)* Test mini training: See QUICKSTART_APE_DATA.md
2. Move to cloud GPU for full training
3. See CLOUD_GPU_TRAINING_APE.md for production settings

---

## If Tests Fail

### Common issues:

**"pydicom not installed"**
```bash
pip install pydicom
```

**"Data directory not found"**
- Check path: `ls /Users/kuntalkokate/Desktop/LLM_agents_projects/dataset`
- Should see `APE/` and `non-APE/` folders
- Update path in test script if different

**"Could not load patient"**
- Check zip files: `ls /Users/kuntalkokate/Desktop/LLM_agents_projects/dataset/APE/`
- Verify files aren't corrupted
- Try extracting one manually

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version (need 3.8+)
python --version
```

---

## After Testing

Once tests pass, you have two options:

### Option A: Quick Local Training Test (CPU, slow but verifies training)

See detailed instructions in `QUICKSTART_APE_DATA.md` Step 5-6

### Option B: Go Straight to Cloud GPU (Recommended)

1. Upload code to cloud GPU instance
2. Use full training config (see CLOUD_GPU_TRAINING_APE.md)
3. Enable pretrained weights for 6x speedup
4. Train for 50+ epochs

---

## Key Files

- `test_ape_data_loading.py` - Main test script
- `data/ape_dataset.py` - APE dataset loader (handles DICOM)
- `RUN_TESTS.sh` - Automated test runner
- `QUICKSTART_APE_DATA.md` - Detailed testing guide
- `config/test_ape_config.yaml` - Minimal config for testing

---

## Questions?

- **What data format?** DICOM medical CT scans in zip files
- **How much data needed?** Your 6 samples are enough for testing; download full dataset for training
- **Training time?** CPU: hours, GPU: ~1 day with pretrained weights
- **Cost?** Testing is free (local), cloud GPU varies by provider

---

**Ready?** Run `./RUN_TESTS.sh` and see if everything works!
