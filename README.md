# RCT Checker

![Tests](https://img.shields.io/github/actions/workflow/status/RickardKarl/rct-checker/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![LLM](https://img.shields.io/badge/LLM-powered-ff69b4)
[![Follow on Bluesky](https://img.shields.io/badge/Bluesky-Follow-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/rickardkarlsson.bsky.social)

A tool for detecting statistical anomalies in data from randomized controlled trials (RCTs) presented in medical research papers using automated data extraction and statistical testing.

Medical research papers typically report baseline characteristics of each treatment group in Table 1. Under proper randomization, these summary statistics should follow predictable distributions. RCT Checker extracts Table 1 from PDFs using LLMs and tests whether the reported statistics deviate from what is expected under randomization. It is intended as a tool for metascience and sanity checking in medical research.

The tool supports using a model from OpenAI (in which case an API key is required) or can be run with an open-weight LLM from HuggingFace.

Note: RCT Checker flags statistical irregularities consistent with deviations from ideal randomization.  
**It does not prove fraud or misconduct.** Results should be interpreted alongside study design, sample size, and domain knowledge.


## How it works

1. **Extract** -- Point at a PDF (file, folder, or URL) and an LLM extracts Table 1 data into structured JSON
2. **Analyze** -- Run chi-squared variance tests (continuous variables) and Fisher's exact tests (categorical variables) to detect statistical anomalies
3. **Report** -- Generate Markdown reports with visualizations

### Quick start

Generate a Markdown report highlighting statistically anomalous baseline characteristics for a folder of PDF files.

```bash
python main.py extract --pdf path/to/pdf_folder
python main.py analyze --report --report-plots
```

### Example report

See [report-example.pdf](report-example.pdf) for a sample report.

## Methodology

The statistical analysis exploits the fact that proper randomization produces predictable distributions of baseline summary statistics across treatment groups. This approach is directly inspired by the methodology of Dr. John Carlisle in [this paper](https://pubmed.ncbi.nlm.nih.gov/22404311/).

Consider the following example: in a randomized controlled trial where individuals are assigned to control or treatment by a coin flip, a categorical variable like sex should have roughly the same distribution in both groups. These kinds of summary counts (or, in the case of continuous variables such as age, means and standard deviations) are typically reported in Table 1 of medical papers. We can extract this information and test whether the observed distributions deviate from what is expected under proper randomization. Deviations that are too large suggest implausible imbalances; deviations that are too small suggest suspiciously perfect balance. Finally, we combine evidence across all variables into a single p-value using Fisher's method, representing the probability of observing results at least as extreme under the null hypothesis of proper randomization.

Below, we describe the specific tests applied to continuous and categorical variables.

### Continuous variables

For each continuous variable reported with a mean and SD (or 95% CI, from which SD is recovered), a z-score is computed per group:

```
z = (group_mean - population_mean) / SEM
```

where `population_mean` is the weighted average across groups and `SEM = SD / sqrt(n)`. Under proper randomization, these z-scores should be approximately standard normal. A **chi-squared variance test** then checks whether the observed variance of all z-scores across all continuous variables deviates from 1. 

Note: the test assumes independence of the z-scores, which cannot be guaranteed since baseline variables are often correlated. This could make the test less conservative (i.e. more false positives). The impact is likely minor when correlations are weak, but accounting for this will be addressed in future iterations. If needed, it is possible to skip this test using the `--skip-cont flag` when running the analysis.

### Categorical variables

For each categorical variable with reported counts, a **Fisher's exact test** is run on the corresponding contingency table (counts vs. group membership). This tests whether the observed distribution of categories across groups is consistent with random allocation. For tables with more than two groups, a Monte Carlo approximation is used.

### Combining statistical tests

All p-values (one from the chi-squared variance test plus one per categorical variable) are aggregated into a single combined p-value using **Fisher's method**. Under the null hypothesis of proper randomization, this combined p-value represents the probability of observing test results at least as extreme as those obtained.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/RickardKarl/rct-checker.git
cd rct-checker
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

For development (linting, testing):

```bash
pip install -e ".[dev]"
pre-commit install
```

For local LLM support via HuggingFace:

```bash
pip install -e ".[huggingface]"
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required for the default `openai` backend) |

### Optional

| Variable | Default | Description |
|---|---|---|
| `RCT_CHECKER_OPENAI_MODEL` | `gpt-5-mini` | OpenAI model to use |
| `RCT_CHECKER_HUGGINGFACE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model to use |
| `RCT_CHECKER_DB_PATH` | `data/paper_database.sqlite` | Path to the SQLite database file |

## Usage

### Extract Table 1 from PDFs

```bash
# Single PDF
python main.py extract --pdf paper.pdf

# Directory of PDFs
python main.py extract --pdf papers/

# URL
python main.py extract --pdf https://example.com/paper.pdf

# Force re-extraction
python main.py extract --pdf paper.pdf --force

# Use HuggingFace backend instead of OpenAI
python main.py extract --pdf paper.pdf --llm-backend huggingface
```

### List extractions

```bash
# List all extractions
python main.py list

# Filter by status
python main.py list --status success
python main.py list --status failed
```

### Analyze extracted data

```bash
# Analyze all successful extractions and print results in terminal
python main.py analyze

# Analyze a specific extraction by ID
python main.py analyze --id 5

# Generate a report (add plots with optional flag --report-plots)
python main.py analyze --report --report-plots


# Skip continuous or categorical variable analysis (can be helpful for sensitivity checks)
python main.py analyze --skip-cont
python main.py analyze --skip-cat
```

## Development

### Running tests

```bash
pytest -v
```

### Linting and formatting

```bash
ruff check --fix .
black .
```

## Contributing

Contributions are welcome. Please run tests and linters before submitting a PR.

## Citation

If you use RCT Checker in academic work, please cite:

> Rickard Karlsson. *RCT Checker*. GitHub, 2026.
