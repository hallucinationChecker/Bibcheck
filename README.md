# Bibcheck
Script to check bibliographies for correctness.  **This script should only be used as a first pass.**  Many correct references will not be found, particularly references to github repositories or webpages.  Any red output should be verified externally, as it may not be available on the sources checked by the script, or there may be a small typo within the citation.  All other colored output can be verified as correct citations within the provided output.

## Running the script
Currently, you can only run the script for papers with either IEEE or ACM references.

### Checking IEEE Papers:
`python3 bibcheck.py paper.pdf ieee`

### Checking ACM Papers:
`python3 bibcheck.py paper.pdf acm`

## Understanding the output
For each citation, red output means a match was not found.  All other colors indicate a potential match.  The output consists of the following.
1. Titles : The input title from the provided citation is first printed, followed by the matched title.  If these are green, the script believes the titles match but please verify.
2. Authors : The input author(s) are printed next, followed by the matched paper's authors.  If these are cyan, the script believes the authors match, but please verify.  If these are orange, the script is unable to determine that the author's names match, likely due to small discrepencies such as diacritics.
3. Full input and output : The input information and the matched output are printed last.  A blue color indicates a match, while red indicates no match was found.

## Acknowledgements
ChatGPT 5 generated large portions of this code.
