# Step 1: parse "antigenic tables" from xlsx:
python parse_and_clean_pdfs/clean_xlsx.py reports_xlsx reports_clean h3n2
python parse_and_clean_pdfs/clean_xlsx.py reports_xlsx reports_clean h1n1

# Step 2: clean data and extract vaccine-virus pairs from tables
mkdir hi_processed
python parse_and_clean_pdfs/build_pairs.py reports_clean/h1n1 hi_processed/a_h1n1_pairs.csv
python parse_and_clean_pdfs/build_pairs.py reports_clean/h3n2 hi_processed/a_h3n2_pairs.csv