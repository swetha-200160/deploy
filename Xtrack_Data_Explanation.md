# Xtrack Data (new).xlsx — Full Data Explanation

## Overview

This Excel file contains **audit and compliance management data** from the **Xtrack CCMS (Continuous Control Monitoring System)** platform. It is used by multiple companies to monitor procurement processes (P2P) and banking operations, and to track audit observations and their resolution status.

- **File:** `Xtrack Data (new).xlsx`
- **Total Sheets:** 5
- **Companies Covered:** A2Z Marketing LTD, A2Z Trading CO, TravelPoint
- **Platform:** Xtrack / CCMS

---

## Sheet 1: `Report Summary`

**Rows:** 65 | **Columns:** 12

### What It Is
A master index/summary of all audit reports generated in the system. Each row represents one audit report entry with its metadata and status.

### Columns

| Column | Description | Unique Values / Notes |
|---|---|---|
| `Company Name` | Company being audited | A2Z Marketing LTD, A2Z Trading CO, TravelPoint |
| `Entity Name` | Department within the company | Finance, Purchase, HR Dept, IT |
| `Business Process` | Process area being audited | P2P, Banking, Finance, Payroll, AMS, ITSEC |
| `Project Name` | Audit project name | Project, CCMS, POC_Mazoon |
| `Action Status` | Current resolution status | Bussiness Pending, Completed, Validation Pending |
| `Severity` | Risk severity level | High, Medium, Low |
| `Report Name` | Name of the specific report | 46 unique report types |
| `Report Summary` | Description / observation text | Free text field |
| `Report ID` | Unique numeric ID for the report | 55 unique IDs |
| `Created By` | Auditor who created the report | Arshitha, Sheelan, Shifa, Siva, Admin, Demo User |
| `Created Date` | Date report was created | Range: 2020-02-04 to 2026-04-24 |
| `Count` | Number of exceptions in that report | Numeric count |

### Key Observations
- Three companies span six business processes
- Most reports are in **Bussiness Pending** status (not yet resolved)
- Reports date back to 2020 with recent ones up to April 2026
- This sheet acts as the **parent index** for all other sheets

---

## Sheet 2: `missing GRN 67222`

**Rows:** 101 | **Columns:** 51

### What It Is
Audit exceptions where a **Goods Receipt Note (GRN) is missing** in the Purchase-to-Pay (P2P) process. This is part of a **3-Way Match** audit — comparing Purchase Order (PO) vs GRN vs Invoice to ensure all three align.

A missing GRN means goods were invoiced but there is no record of the goods being physically received.

### Column Groups

#### Audit Metadata (Columns 1–16)

| Column | Description |
|---|---|
| `Tran Ref No` | Unique transaction reference (e.g., AD2026000038) |
| `Created By` | Auditor who logged the observation |
| `Name of Observation` | Type of issue (Missing GRN) |
| `Observation Desc` | Detailed description of the observation |
| `ActionPlan` | Action plan entered by the responsible person |
| `DueDate` | Deadline for resolution |
| `Recommendation` | Auditor's recommendation |
| `Responsible` | Person responsible for resolving the issue |
| `Date` | Date the observation was logged |
| `Severity` | Risk level: High or Medium |
| `Remarks` | Additional remarks |
| `Assignment` | Assignment details |
| `Company` | A2Z Trading CO (all records) |
| `Business Process` | Finance (all records) |
| `Owner` | Owner of the process |
| `ActionStatus` | Action Open (all records — none resolved) |

#### Purchase Order (PO) Data (Columns 17–32)

| Column | Description |
|---|---|
| `PO_NO_C` | Purchase Order number (e.g., PO001006) |
| `PO_LINE_C` | Line item number within the PO |
| `PO_DATE` | Date the PO was raised |
| `JOIN_KEY` | Composite key: PO_NO + PO_LINE (e.g., PO001006-11) |
| `PO_AMOUNT` | Total PO amount |
| `PO_QTY` | Quantity ordered |
| `PO_UOM_C` | Unit of measure code |
| `PO_UNIT_PRICE` | Unit price in the PO |
| `VENDOR_ID` | Vendor identifier code (e.g., V0002) |
| `VENDOR_C` | Vendor name |
| `PO_CURRENCY` | Currency: AED, USD, EUR |
| `BUYER` | Buyer responsible: Buyer-A, Buyer-B, Buyer-C, Buyer-D |
| `DELIVERY_DATE` | Expected delivery date |
| `MATERIAL_DESC` | Material/product description |
| `MATERIAL_ID` | Material identifier code |
| `PAYMENT_TERMS` | Payment terms: NET30, NET45, NET60, COD, ADV |
| `PLANT` | Warehouse/plant location: AJM1, AUH1, DXB1, SHJ1 |

#### GRN Data (Columns 33–37)

| Column | Description |
|---|---|
| `GRN_COUNT` | Number of GRN records found (0 = missing GRN) |
| `GRN_DATE_MAX` | Latest GRN date |
| `GRN_DATE_MIN` | Earliest GRN date |
| `GRN_JOIN_KEY` | Join key linking GRN to PO line |
| `GRN_QTY_SUM` | Total quantity received per GRN |

#### Invoice Data (Columns 38–44)

| Column | Description |
|---|---|
| `INV_AMT_SUM` | Total invoice amount |
| `INV_COUNT` | Number of invoices |
| `INV_DATE_MIN` | Earliest invoice date |
| `INV_JOIN_KEY` | Join key linking invoice to PO line |
| `INV_QTY_SUM` | Total quantity invoiced |
| `INV_UNITPRICE_AVG` | Average invoice unit price |

#### Exception Flags (Columns 45–51)

| Column | Description | Flagged Count (out of 100) |
|---|---|---|
| `EXC_MISSING_GRN` | 1 = GRN is missing | 19 |
| `EXC_MISSING_INV` | 1 = Invoice is missing | 1 |
| `EXC_PARTIAL_RECEIPT` | 1 = Partial goods received | 19 |
| `EXC_OVER_RECEIPT` | 1 = More goods received than ordered | 0 |
| `EXC_QTY_VAR` | 1 = Quantity mismatch | 2 |
| `PRICE_VAR_PCT` | % price variance (invoice vs PO) | Numeric |
| `EXC_PRICE_VAR` | 1 = Price variance detected | 4 |

### Key Observations
- All 100 records have `ActionStatus = Action Open` — no issues resolved yet
- 13 unique vendors (some with name duplicates, e.g., DELTA-SERVICES FZC vs F.Z.C.)
- 4 plants: AJM1 (Ajman), AUH1 (Abu Dhabi), DXB1 (Dubai), SHJ1 (Sharjah)
- Materials include: Laptop, Safety Gloves, Motor Oil, Spare Parts, Chemicals, Medical Syringes, etc.

---

## Sheet 3: `Price Variance 67227`

**Rows:** 94 | **Columns:** 56

### What It Is
Audit exceptions where the **invoice unit price significantly differs from the PO unit price** — a price variance in the 3-Way Match process. Indicates potential overcharging, undercharging, or fraud.

### Column Structure
Same as the Missing GRN sheet (columns 1–51) plus 5 additional text-flag columns:

| Column | Description |
|---|---|
| `MISSING_GRN` | T/F — GRN is missing |
| `MISSING_INV` | T/F — Invoice is missing |
| `PARTIAL_RECEIPT` | T/F — Partial receipt of goods |
| `OVER_RECEIPT` | T/F — Over-receipt of goods |
| `QTY_VAR` | T/F — Quantity variance exists |

### Key Data Points

| Field | Details |
|---|---|
| `Tran Ref No` | All records = CM2026000053 (same audit run) |
| `PRICE_VAR_PCT` | Ranges from -1.0 (100% deficit) to +0.97 (97% overcharge) |
| `EXC_PRICE_VAR` | 1 = price variance confirmed |
| Report code | `R990_3WayMatch` in Observation Desc |

### Key Observations
- A `PRICE_VAR_PCT` of `-1.0` means invoice amount is zero (missing invoice)
- A positive value means the invoice charged more than the PO price
- A negative value means the invoice charged less than the PO price
- All flagged records are from the same automated audit run on 2026-04-22

---

## Sheet 4: `Cheque Clearing 67318`

**Rows:** 6,033 | **Columns:** 33

### What It Is
The largest sheet. Contains **bank cheque clearing data** from branches across Oman. Tracks whether each cheque was successfully cleared or rejected, along with the reason and processing details.

### Columns

| Column | Description |
|---|---|
| `Tran Ref No` | Audit transaction reference (CM2026000065) |
| `Created By` | Arshitha (auditor) |
| `Name of Observation` | Cheque Clearing |
| `Observation Desc` | R003_Cheque_Clearing |
| `Date` | Audit log date: 2026-04-24 |
| `ChequeNumber` | Unique cheque identifier (e.g., CHQ4732588) |
| `Amount` | Cheque face value |
| `Cheque_Date` | Date cheque was issued |
| `UserId` | Bank teller/user ID (numeric) |
| `Status` | Processing status code: 0 to 7 |
| `Branch` | Bank branch name |
| `Region` | Geographic region in Oman |
| `UserName` | Name of the bank teller/processor |
| `Reason` | Outcome reason (Cleared, Account Closed, Invalid Signature, etc.) |
| `Cheque_Status` | Final status: **Cleared** or **Rejected** |
| `Processed` | 1 = processed (all records = 1) |
| `Cleared` | 1 = cheque cleared, 0 = not cleared |
| `ClearedAmount` | Amount that was cleared |
| `Rejected` | 1 = cheque rejected, 0 = not rejected |
| `RejectedAmount` | Amount that was rejected |
| `cWeekDay` | Day of week: Mon, Tue, Wed, Thu, Fri, Sat, Sun |
| `cPeriod` | Month-Year period (e.g., Dec-2024) |

### Geographic Coverage (Oman Regions)

| Region |
|---|
| Muscat |
| Al Batinah North |
| Al Batinah South |
| Dhofar |
| Al Buraimi |
| Ad Dakhiliyah |
| Al Dhahirah |
| Al Wusta |
| Ash Sharqiyah North |
| Ash Sharqiyah South |
| Al Sharqiyah South |

### Key Observations
- 6,033 cheque records — the most data-rich sheet
- Two outcomes only: **Cleared** or **Rejected**
- Rejection reasons include: Account Closed, Invalid Signature, and others
- Covers all days of the week and multiple months
- All records linked to a single audit report (CM2026000065) from April 2026

---

## Sheet 5: `Quantity Variance 67226`

**Rows:** 136 | **Columns:** 55

### What It Is
Audit exceptions where the **quantity received (GRN) or invoiced differs from the quantity ordered (PO)** — quantity variance in the 3-Way Match process. Indicates goods were over-delivered, under-delivered, or incorrectly invoiced.

### Column Structure
Same as the Price Variance sheet (columns 1–51) plus 4 additional text-flag columns:

| Column | Description |
|---|---|
| `MISSING_GRN` | T/F — GRN is missing |
| `MISSING_INV` | T/F — Invoice is missing |
| `PARTIAL_RECEIPT` | T/F — Partial receipt |
| `OVER_RECEIPT` | T/F — Over-receipt |

### Key Data Points

| Field | Details |
|---|---|
| `Tran Ref No` | All records = CM2026000053 (same audit run as Price Variance) |
| `EXC_QTY_VAR` | 1 = quantity variance confirmed |
| `INV_QTY_SUM` vs `PO_QTY` | Comparison reveals over/under delivery |
| Report code | `R990_3WayMatch` |

### Key Observations
- Same audit run as Price Variance sheet (CM2026000053, 2026-04-22)
- Records can show both quantity AND price variance simultaneously
- Some PO lines appear in both Price Variance and Quantity Variance sheets (e.g., PO001002-49)
- Quantity variances can be small (1 unit) or large (20+ units)

---

## Relationship Between Sheets

```
Report Summary
    └── Master index of all reports
        ├── Report ID 67222 → Sheet: missing GRN 67222
        ├── Report ID 67226 → Sheet: Quantity Variance 67226
        ├── Report ID 67227 → Sheet: Price Variance 67227
        └── Report ID 67318 → Sheet: Cheque Clearing 67318
```

### P2P Sheets Are Strongly Related

The three procurement sheets (**Missing GRN**, **Price Variance**, **Quantity Variance**) are different **filter views** of the same underlying 3-Way Match dataset:

- Same `Tran Ref No`: CM2026000053
- Same PO numbers appear across sheets (e.g., PO001002-49, PO001006-90)
- Same vendors, materials, buyers, plants
- Same exception flag columns — each sheet highlights a different type of anomaly
- Same audit run date: 2026-04-22

### Cheque Clearing Is Separately Related

- Different business process (Banking vs P2P)
- Different `Tran Ref No`: CM2026000065
- Different region (Oman banking vs UAE procurement)
- Shares the same Xtrack/CCMS platform and audit framework

---

## Data Domain Summary

| Sheet | Business Domain | Geography | Volume | Key Anomaly |
|---|---|---|---|---|
| Report Summary | Audit Management | UAE + Oman | 65 rows | Status tracking |
| missing GRN 67222 | Procurement / P2P | UAE (AJM, AUH, DXB, SHJ) | 101 rows | GRN not found |
| Price Variance 67227 | Procurement / P2P | UAE | 94 rows | Invoice price ≠ PO price |
| Cheque Clearing 67318 | Banking | Oman (11 regions) | 6,033 rows | Cheque rejected |
| Quantity Variance 67226 | Procurement / P2P | UAE | 136 rows | Qty received ≠ Qty ordered |

---

## Conclusion

This dataset is an **end-to-end audit trail** for the Xtrack CCMS platform used by A2Z group companies and TravelPoint. It covers:

1. **Procurement Integrity (3-Way Match):** Detects financial risks where PO, GRN, and Invoice do not align — covering missing receipts, price overcharges, and quantity mismatches in UAE operations.
2. **Banking Control (Cheque Clearing):** Monitors cheque clearing outcomes across all Oman regions to detect failed transactions and processing patterns.
3. **Audit Governance:** The Report Summary tracks all observations, assigns severity, and monitors resolution status across auditors and companies.

All sheets are interconnected through the Xtrack platform's report ID system and share a common audit observation structure (Tran Ref No, Created By, Severity, ActionStatus).
