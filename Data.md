#### Data Overview
In this competition, participants will extract all research data referenced in a scientific paper (by their identifier) and classify it based on its context as a primary or secondary citation.

#### Paper and Dataset Identifiers
Each object (paper and dataset) has a unique, persistent identifier to represent it. In this competition there will be two types:

DOIs are used for all papers and some datasets. They take the following form: https://doi.org/[prefix]/[suffix]. Examples:
https://doi.org/10.1371/journal.pone.0303785
https://doi.org/10.5061/dryad.r6nq870
Accession IDs are used for some datasets. They vary in form by individual data repository where the data live. Examples:
"GSE12345" (Gene Expression Omnibus dataset)
“PDB 1Y2T” (Protein Data Bank dataset)
"E-MEXP-568" (ArrayExpress dataset)
Files
train/{PDF,XML} - the training articles, in PDF and XML format
IMPORTANT: Not all PDF articles have a corresponding XML file (approx. 75% do)
test/{PDF,XML} - the test articles, in PDF and XML format
The rerun test dataset has approximately 2,600 articles.
train_labels.csv - labels for the training articles
article_id - research paper DOI, which will be located in the full text of the paper
dataset_id - the dataset identifier and citation type in the paper.
type - citation type
Primary - raw or processed data generated as part of this paper, specifically for this study
Secondary - raw or processed data derived or reused from existing records or published data
sample_submission.csv - a sample submission file in the correct format
The full text of the scientific papers were downloaded in PDF & XML from at: Europe PMC open access subset.

#### Data Citation Mining Examples
To illustrate how research data are mentioned in the scientific literature, here are some examples:
Note: in the text, the dataset identifier may appear with or without the 'https://doi.org' stem.

Paper: https://doi.org/10.1098/rspb.2016.1151
Data: https://doi.org/10.5061/dryad.6m3n9
In-text span: "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9."
Citation type: Primary
Paper: https://doi.org/10.1098/rspb.2018.1563
Data: https://doi.org/10.5061/dryad.c394c12
In-text span: "Phenotypic data and gene sequences are available from the Dryad Digital Repository: http://dx.doi.org/10.5061/dryad.c394c12"
Citation type: Primary
Paper: https://doi.org/10.1534/genetics.119.302868
Data: https://doi.org/10.25386/genetics.11365982
In-text span: "The authors state that all data necessary for confirming the conclusions presented in the article are represented fully within the article. Supplemental material available at figshare: https://doi.org/10.25386/genetics.11365982."
Citation type: Primary
Paper: https://doi.org/10.1038/sdata.2014.33
Data: GSE37569, GSE45042, GSE28166
In-text span: "Primary data for Agilent and Affymetrix microarray experiments are available at the NCBI Gene Expression Omnibus (GEO, http://www.ncbi.nlm.nih.gov/geo/) under the accession numbers GSE37569, GSE45042 , GSE28166"
Citation type: Primary
Paper: https://doi.org/10.12688/wellcomeopenres.15142.1
Data: pdb 5yfp
In-text span: “Figure 1. Evolution and structure of the exocyst. A) Cartoon representing the major supergroups, which are referred to in the text. The inferred position of the last eukaryotic common ancestor (LECA) is indicated and the supergroups are colour coordinated with all other figures. B) Structure of trypanosome Exo99, modelled using Phyre2 (intensive mode). The model for the WD40/b-propeller (blue) is likely highly accurate. The respective orientations of the a-helical regions may form a solenoid or similar, but due to a lack of confidence in the disordered linker regions this is highly speculative. C and D) Structure of the Saccharomyces cerevisiae exocyst holomeric octameric complex. In C the cryoEM map (at level 0.100) is shown and in D, the fit for all eight subunits (pdb 5yfp). Colours for subunits are shown as a key, and the orientation of the cryoEM and fit are the same for C and D. All structural images were modelled by the authors from PDB using UCSF Chimera.”
Citation type: Secondary
Paper: https://doi.org/10.3389/fimmu.2021.690817
Data: E-MTAB-10217, PRJE43395
In-text span: “The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.”
Citation type: Secondary