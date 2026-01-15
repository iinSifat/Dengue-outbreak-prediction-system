# 🚀 INSTALLATION & SETUP GUIDE

## Dengue Outbreak Prediction System

### Prerequisites
- Windows OS (You're using Windows)
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for installing packages)

---

## Step-by-Step Installation

### Step 1: Check Python Installation

Open PowerShell and run:
```powershell
python --version
```

You should see: `Python 3.8.x` or higher

If not installed, download from: https://www.python.org/downloads/

---

### Step 2: Navigate to Project Directory

```powershell
cd "e:\Academic Study\UIU trimester-7\AI LAB\Ai project\dengue_prediction_project"
```

---

### Step 3: Install Required Packages

```powershell
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- matplotlib (plotting)
- seaborn (visualization)
- streamlit (web interface)
- statsmodels (statistical models)

**Installation may take 2-5 minutes**

---

### Step 4: Verify Installation

Run the demo script:
```powershell
python demo.py
```

You should see:
- Data loading messages
- Model training progress
- Performance metrics
- Success confirmation

**If you see "ALL TESTS PASSED! ✓" - you're ready!**

---

### Step 5: Launch the Application

```powershell
streamlit run app.py
```

The browser will automatically open to: http://localhost:8501

**If browser doesn't open, manually go to that URL**

---

## Troubleshooting

### Problem: "python is not recognized"
**Solution:** Add Python to PATH or use full path:
```powershell
C:\Python38\python.exe demo.py
```

### Problem: "pip install fails"
**Solution:** Try with --user flag:
```powershell
pip install --user -r requirements.txt
```

### Problem: "Module not found"
**Solution:** Ensure you're in the correct directory:
```powershell
cd "e:\Academic Study\UIU trimester-7\AI LAB\Ai project\dengue_prediction_project"
```

### Problem: "Streamlit not opening"
**Solution:** Manually open browser and go to:
```
http://localhost:8501
```

### Problem: "Permission denied"
**Solution:** Run PowerShell as Administrator

---

## Quick Test Commands

### Test 1: Check files exist
```powershell
dir data
dir models
dir utils
```

### Test 2: Test Python imports
```powershell
python -c "import pandas; import numpy; import sklearn; print('All imports OK!')"
```

### Test 3: Run demo
```powershell
python demo.py
```

### Test 4: Start app
```powershell
streamlit run app.py
```

---

## What to Expect

### After running demo.py:
✅ Console output showing:
- Data loading progress
- Model training steps
- Performance metrics
- Prediction examples
- Success message

### After running streamlit run app.py:
✅ Browser opens with:
- Modern web interface
- Interactive tabs
- Charts and graphs
- Prediction controls
- Risk analysis

---

## Using the Application

### 1. Overview Tab
- View dataset statistics
- See historical trends
- Check monthly patterns

### 2. Predictions Tab
- Select prediction horizon (1-24 weeks)
- View model performance
- See future forecasts
- Check risk alerts

### 3. Risk Zones Tab
- Explore cluster analysis
- View risk distributions
- Check state transitions

### 4. Analysis Tab
- Seasonal patterns
- Error analysis
- Statistical summaries

### 5. About Tab
- System documentation
- Algorithm explanations
- Usage guidelines

---

## Stopping the Application

### To stop Streamlit:
Press `Ctrl + C` in the PowerShell window

---

## File Structure Verification

Your project should have this structure:

```
dengue_prediction_project/
│
├── data/
│   ├── dengue_cases.csv       ✅ Check exists
│   └── weather_data.csv        ✅ Check exists
│
├── models/
│   ├── __init__.py            ✅ Check exists
│   ├── regression.py           ✅ Check exists
│   ├── clustering.py           ✅ Check exists
│   └── markov.py               ✅ Check exists
│
├── utils/
│   ├── __init__.py            ✅ Check exists
│   ├── preprocessing.py        ✅ Check exists
│   └── visualization.py        ✅ Check exists
│
├── app.py                      ✅ Check exists
├── demo.py                     ✅ Check exists
├── requirements.txt            ✅ Check exists
├── README.md                   ✅ Check exists
├── QUICKSTART.md              ✅ Check exists
├── PROJECT_SUMMARY.md         ✅ Check exists
└── INSTALLATION.md            ✅ Check exists (this file)
```

---

## Common Use Cases

### For Demonstration:
```powershell
streamlit run app.py
```
Navigate through tabs, show predictions

### For Testing:
```powershell
python demo.py
```
Verify all models work

### For Development:
Open files in VS Code or any editor
Modify and test

---

## Performance Tips

### Faster Loading:
- First run takes longer (model training)
- Subsequent runs use caching
- Streamlit caches data and models

### Better Visualization:
- Use full-screen browser mode
- Close other tabs for better performance
- Recommended: Chrome or Edge browser

---

## System Requirements

### Minimum:
- 4 GB RAM
- 2 GB free disk space
- 1 GHz processor

### Recommended:
- 8 GB RAM
- 5 GB free disk space
- Multi-core processor

---

## Next Steps After Installation

1. ✅ Run demo.py to verify
2. ✅ Launch app.py for UI
3. ✅ Explore all tabs
4. ✅ Try different predictions
5. ✅ Read PROJECT_SUMMARY.md
6. ✅ Prepare for presentation

---

## Getting Help

### If Issues Persist:

1. **Check Python version**
   ```powershell
   python --version
   ```

2. **Reinstall packages**
   ```powershell
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

3. **Check file paths**
   Ensure you're in correct directory

4. **Review error messages**
   Read the full error output

---

## Success Indicators

### ✅ Installation Successful If:
- demo.py runs without errors
- All tests pass
- Streamlit opens in browser
- UI loads completely
- Predictions work

### ✅ Ready for Presentation If:
- Can launch app quickly
- All tabs display correctly
- Predictions generate
- Charts render properly
- No error messages

---

## Final Checklist

Before your presentation:

- [ ] Python installed and working
- [ ] All packages installed
- [ ] demo.py runs successfully
- [ ] Streamlit app launches
- [ ] All tabs accessible
- [ ] Predictions working
- [ ] Charts displaying
- [ ] Understand the code
- [ ] Read documentation
- [ ] Practice demo flow

---

## Contact & Support

For technical issues:
- Review README.md
- Check PROJECT_SUMMARY.md
- Examine code comments
- Review error messages

---

## Congratulations! 🎉

If you've completed all steps successfully, you have a fully functional AI system ready for demonstration!

**Your project includes:**
- ✅ 3 ML algorithms
- ✅ Interactive web interface
- ✅ Real dataset
- ✅ Comprehensive documentation
- ✅ Professional code quality

**You're ready to impress!** 🚀

---

**End of Installation Guide**
