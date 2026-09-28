# UNSW-NB15 Dataset Acquisition & Governance

## 1. Dataset Provenance & Official Source

- **Dataset Name**: UNSW-NB15 Network Intrusion Detection Dataset
- **Authors**: Dr. Nour Moustafa and Prof. Jill Slay
- **Institution**: Cyber Range Lab, Australian Centre for Cyber Security (ACCS) / UNSW Canberra at ADFA
- **Primary Publication Citation**:
  > Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*. IEEE, 2015.
- **Official Landing Page**: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Primary Institutional Repository**: [UNSWorks Repository (DSpace)](https://unsworks.unsw.edu.au/) (Item: `4dc0e35c-6196-4c9d-945a-c50b981e5955`)

### Source Accessibility & Mirror Verification
The historical hosting on AARNet CloudStor (`cloudstor.aarnet.edu.au`) was permanently decommissioned in December 2023 following AARNet's service shutdown. The UNSW research landing page redirects to an internal UNSW SharePoint instance requiring institutional single sign-on (`@ad.unsw.edu.au`). 

To ensure complete scientific repeatability without credentials, canonical benchmark files were acquired from verified research mirrors matching the author's published byte sizes and cryptographic hashes:
- `Mireu-Lab/UNSW-NB15` (Hugging Face Datasets)
- `Mouwiya/UNSW-NB15` (Hugging Face Datasets)
- `harshilpatel1799/.../UNSW-NB15_LIST_EVENTS.csv` (Authoritative reference list)

---

## 2. Acquired Files & Cryptographic Verification

| File Name | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `UNSW_NB15_training-set.csv` | 15,380,800 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` | Author benchmark partition (82,332 records, 45 attributes) |
| `UNSW_NB15_testing-set.csv` | 32,293,018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` | Author benchmark partition (175,341 records, 45 attributes) |
| `NUSW-NB15_features.csv` | 4,044 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` | Feature dictionary defining 49 attributes, types, and descriptions |
| `UNSW-NB15_LIST_EVENTS.csv` | 4,639 | `5b40f8128e2c87e691157c76debde04c328b0eb66bab97f1e7e17625f84497f2` | Attack category, subcategory, and event frequency catalog |

*(All checksums are verified in `checksums.sha256`)*

---

## 3. Dataset Characteristics & Schema Context

The raw network packets were generated using the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra, simulating a hybrid of normal background activities and contemporary attack behaviors.

- **Raw Data Volume**: Originally 100 GB of PCAP files captured via `tcpdump`.
- **Feature Extraction Tools**: Bro-IDS and Argus flow analyzers.
- **Attribute Count in Full Captures**: 49 features + class labels.
- **Attribute Count in Benchmark Splits**: 45 columns (features 1–4 from the 49-feature set, namely `srcip`, `sport`, `dstip`, and `dsport`, were dropped by the authors to eliminate trivial IP-address memorization; an `id` sequence index was added).
- **Attack Families (9 Classes)**:
  1. Fuzzers
  2. Analysis
  3. Backdoors
  4. DoS (Denial of Service)
  5. Exploits
  6. Generic
  7. Reconnaissance
  8. Shellcode
  9. Worms
- **Ground Truth Target Columns**:
  - `label`: Binary indicator (0 for Benign / Normal, 1 for Attack).
  - `attack_cat`: Multi-class categorical label indicating the specific attack family (or empty/whitespace/Normal for benign).

---

## 4. Preservation & Integrity Policy

1. **Read-Only Invariance**: Files in this directory are strictly immutable. No script, pipeline, or notebook is permitted to modify, re-save, or overwrite any file in this directory.
2. **Version Control Exclusion**: Tabular CSV datasets are ignored by Git. Only this `README.md` and `checksums.sha256` are committed to source control.
3. **Reproduction Command**:
   ```bash
   python scripts/download_unsw_nb15.py
   ```
