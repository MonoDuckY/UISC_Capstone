# Naming Convention in Data Folder

- **raw data**: No constraints (e.g., `patient_001.png`)
- **processed data**: `[raw data name]_processed.[ext]` (e.g., `patient_001_processed.png`)
- **labeled data**: `[raw data name]_labeled.[ext]` (e.g., `patient_001_labeled.png`)
- **annotation**: `[raw data name]_annotation.xml` (e.g., `patient_001_annotation.xml`)

## Notes
- Annotation files must be in `.xml` format.
- Image files should be in `.png`, `.jpg`, or `.jpeg` format.