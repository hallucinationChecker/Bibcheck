# Bibcheck
Script to check bibliographies for correctness.  **This script should only be used as a first pass.**  Many correct references will not be found, particularly references to github repositories or webpages.  Any red output should be verified externally, as it may not be available on the sources checked by the script, or there may be a small typo within the citation.  All other colored output can be verified as correct citations within the provided output.

## Running the script
Currently, you can only run the script for papers with either IEEE or ACM references.

### Checking IEEE Papers:
`python3 bibcheck.py paper.pdf ieee`

### Checking ACM Papers:
`python3 bibcheck.py paper.pdf acm`

## Understanding the output
For a successful match, the output consists of the following.
1. Titles : The input title from the provided citation is first printed, followed by the matched title.  If these are green, the script believes the titles match but please verify.  If they are red, the titles are not a perfect match, but close enough to assume they should be; likely there is a typo.
2. Authors : The input author(s) are printed next, followed by the matched paper's authors.  If these are green, the script believes the authors match, but please verify.  If these are red, the script is unable to determine that the author's names match, likely due to small discrepencies such as diacritics, or missing one of the authors.  However, at least one author should be correct.
3. Full input and output : The input information and the matched output are printed last.  A blue color indicates a match.

For a failed match, the output consists only of a red json dump.  This will contain the closest match from crossref, openalex, arxiv, and googlebooks.  Multiple small discrepancies can result in this to happen.  **If no output titles are a reasonable match, you will need to check the citation by hand.**  This script cannot verify websites, which are common for NVIDIA and AMD citations.  The script also cannot verify github code repository citations. 

## Acknowledgements
ChatGPT 5 generated large portions of this code.
