# Resume Job-Category Classifier

This project trains one supervised machine-learning classifier to predict a job-category label from resume text. It does not compare a resume with a job description, score candidate suitability, rank applicants, or make hiring decisions.

## Dataset

The project uses the public [Resume Dataset on Hugging Face](https://huggingface.co/datasets/opensporks/resumes). It contains 2,484 resumes and 24 categories. The included `data/Resume.csv` is a smaller working copy with the two columns used by the model: `Resume_str` (input text) and `Category` (the label to predict). It omits the unused ID and HTML columns and replaces detected email addresses, web links, and phone-like strings. The dataset card says the examples were collected from LiveCareer and lists the license as CC0 1.0.

The model's labels describe the category associated with each example resume. They do not indicate whether an applicant is qualified. The dataset is small and category counts vary, so results should be treated as a classroom demonstration.

## Libraries

- pandas loads and organizes the CSV.
- NumPy calculates the held-out accuracy.
- matplotlib saves dataset and evaluation charts.
- scikit-learn creates TF-IDF text features, splits the data, trains Logistic Regression, and evaluates predictions.

TF-IDF converts words and short phrases into numerical features. Logistic Regression is the only classifier used.

## Run

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install the libraries with `python -m pip install -r requirements.txt`.
4. Keep the downloaded file at `data/Resume.csv`.
5. Run `python resume_role_classifier.py`.

The script trains on 80% of the examples and evaluates on the remaining 20%, using a stratified split with a fixed random seed. It saves metrics, per-category scores, a confusion matrix, and matplotlib charts in `results/`.

## Measured result

With the included dataset and seed 42, the held-out split contained 497 resumes. The run measured 64.4% accuracy, 57.2% macro F1, and 62.1% weighted F1. The model scored F1=0 on BPO and Automobile in this split, which had only four and seven test examples for those categories. These are results from one split, not a promise of future performance.

To try a short, invented example, run:

```text
python resume_role_classifier.py --demo-text "Python, SQL, pandas, and data analysis experience"
```

The displayed class probability is a model estimate, may not be calibrated, and does not measure applicant quality or job fit.

## Data handling

The script replaces detected email addresses, web links, and phone-like strings before vectorizing resume text. These patterns may not catch every identifier. Do not put real applicant resumes into a classroom demo, and do not show personal information from source resumes in slides.

## Source

Dataset page and data card: https://huggingface.co/datasets/opensporks/resumes
