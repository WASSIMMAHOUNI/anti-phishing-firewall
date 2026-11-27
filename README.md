# Anti-Phishing Firewall

Multi-agent email phishing detection pipeline combining header analysis, text classification, and a fusion meta-classifier.

---

## Agent 3 — Meta-Classifier

Agent 3 is the decision-fusion component of the pipeline. It combines the phishing probability scores produced by **Agent 1** (`nb_score`) and **Agent 2** (`text_score`) using a trained Logistic Regression meta-classifier (`agent3_logreg.joblib`) to output the final classification and confidence score.

### Directory & File Structure
* `build_agent3_handoff.py`: Generates and validates the Agent 3 handoff package, configuration, evaluation metrics, and reference examples.
* `agent3_outputs/`:
  * `agent3.py`: Inference module providing the prediction interface for downstream integration.
  * `agent3_logreg.joblib`: Serialized Logistic Regression meta-classifier model.
  * `agent3_config.json`: Decision thresholds, feature names, and runtime configuration.
  * `agent3_meta.json`: Test-set evaluation metrics (Agent 1, Agent 2, and fused Agent 3) and environment metadata.
  * `reference_examples.json`: Verified input/output test cases for sanity-checking predictions.

### Quick Usage

```bash
pip install -r requirements.txt
python build_agent3_handoff.py
