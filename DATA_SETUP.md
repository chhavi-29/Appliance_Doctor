# DATA_SETUP.md — Dataset Setup Guide

## The Invisible Appliance Doctor — Dataset Setup

---

## Option A: Use the Real UCI Dataset (Recommended)

### Step 1 — Download the dataset

1. Open your browser and go to:
   **[https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)**

2. Click the **Download** button on that page.  
   You will receive a ZIP file called something like `individual+household+electric+power+consumption.zip`.

3. Extract the ZIP file. Inside, you will find:
   ```
   household_power_consumption.txt
   ```

### Step 2 — Place the file in the project

Copy (or move) the file into the `data/` folder of this project:

```
Appliance_Doctor/
├── data/
│   └── household_power_consumption.txt   ← put it here
├── src/
├── pages/
├── app.py
└── ...
```

**Windows PowerShell command** (adjust path to match where you extracted the file):
```powershell
Copy-Item "C:\Users\YourName\Downloads\household_power_consumption.txt" `
          "C:\Users\YourName\Desktop\Appliance_Doctor\data\"
```

### Step 3 — Verify the file

Open a terminal in the project folder and run:

```powershell
# Check the file exists and its size (should be ~130 MB)
Get-Item data\household_power_consumption.txt | Select-Object Name, Length

# Preview the first 3 lines
Get-Content data\household_power_consumption.txt -TotalCount 3
```

Expected output of the preview:
```
Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000
```

Key things to verify:
- Fields are separated by **semicolons** (`;`)
- Date format is `dd/mm/yyyy`
- File is approximately **127–132 MB** in size

---

## Option B: Demo Mode (Synthetic Data — No Download Needed)

If you want to explore the dashboard immediately without downloading the real dataset:

1. Launch the app:
   ```powershell
   streamlit run app.py
   ```

2. In the **left sidebar**, toggle **"🎭 Demo Mode (synthetic data)"** to ON.

3. Go to **🧪 Model Lab** and click **Train RBM**.

The app will generate a realistic synthetic dataset automatically. All features of the
dashboard will work, but the patterns are simulated (not from real household data).

> ⚠️ **Note:** Results with synthetic data are for demonstration purposes only.
> For your academic submission, use the real UCI dataset.

---

## File Format Reference

| Property | Value |
|----------|-------|
| Separator | Semicolon (`;`) |
| Missing values | `?` (question mark) |
| Date format | `dd/mm/yyyy` |
| Time format | `hh:mm:ss` |
| Frequency | 1-minute readings |
| Date range | 16 Dec 2006 – 26 Nov 2010 (~47 months) |
| Total rows | ~2,075,259 (excluding header) |
| File size | ~127 MB |

### Columns

| Column | Unit | Description |
|--------|------|-------------|
| `Date` | — | Date of measurement |
| `Time` | — | Time of measurement |
| `Global_active_power` | kilowatt | Total active power consumed |
| `Global_reactive_power` | kilowatt | Reactive power |
| `Voltage` | volt | Household voltage |
| `Global_intensity` | ampere | Current intensity |
| `Sub_metering_1` | watt-hour | Kitchen area energy |
| `Sub_metering_2` | watt-hour | Laundry room energy |
| `Sub_metering_3` | watt-hour | Water heater / AC energy |

---

## Troubleshooting

### "Dataset not found" message in the app

Make sure the file is named **exactly**:
```
household_power_consumption.txt
```
(Some systems may name it `individual+household+electric+power+consumption.txt` — rename it.)

### "Permission denied" when copying

Run your terminal as Administrator, or use File Explorer to drag-and-drop the file.

### File appears to be 0 bytes

The download may have been interrupted. Re-download from the UCI page.

### "UnicodeDecodeError" when loading

The file uses Latin-1 encoding. The `data_loader.py` handles this automatically with
`encoding='latin-1'`. No action needed.

### App runs very slowly

The full dataset has ~2 million rows. The app automatically limits training to
`max_train_days` (default 90 days). Adjust this slider in Model Lab if needed.

---

## Dataset Citation

> Hébrail, G. & Bérard, A. (2012).
> *Individual Household Electric Power Consumption* [Dataset].
> UCI Machine Learning Repository.
> [https://doi.org/10.24432/C58K54](https://doi.org/10.24432/C58K54)

```bibtex
@misc{uci_household_power_2012,
  author       = {H{\'e}brail, Georges and B{\'e}rard, Alice},
  title        = {{Individual Household Electric Power Consumption}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {\url{https://doi.org/10.24432/C58K54}}
}
```
