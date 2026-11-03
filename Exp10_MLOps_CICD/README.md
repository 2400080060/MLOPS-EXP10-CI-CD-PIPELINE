# Experiment 10: Implementing CI/CD Pipelines with GitHub Actions for MLOps

## Aim
Build a simple machine-learning workflow and automate testing, training, artifact storage, and an optional deployment trigger using GitHub Actions.

## Project files
- `train.py` - trains a Random Forest model on the Iris dataset and saves it.
- `test_model.py` - automated tests for model prediction and accuracy.
- `.github/workflows/mlops-ci-cd.yml` - GitHub Actions CI/CD workflow.
- `requirements.txt` - Python dependencies.

## Run in PyCharm (Windows)
1. Open this folder in PyCharm.
2. Create/select a Python 3.10 or 3.11 virtual environment.
3. Open the PyCharm Terminal and run:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run `train.py`. The model is saved to `artifacts/iris_model.joblib`.
5. Run tests:
   ```bash
   pytest -v
   ```

## GitHub Actions setup
1. Create a public or private GitHub repository.
2. In the PyCharm Terminal, run these commands from the project folder:
   ```bash
   git init
   git add .
   git commit -m "Add MLOps CI-CD experiment"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```
3. Open the repository on GitHub and select **Actions**. The workflow runs on push to `main`, pull requests to `main`, or manual dispatch.
4. Open the latest workflow run to inspect the test, training, and artifact-upload steps.

## Optional deployment step
The workflow contains an optional deployment webhook step. Add a deployment provider's webhook URL as a repository secret named `DEPLOY_HOOK_URL`:
**Settings > Secrets and variables > Actions > New repository secret**.
If no secret is configured, the workflow safely skips the deployment trigger. A webhook only triggers an already configured deployment service; it does not create hosting infrastructure by itself.

## Post-lab activities
- Add/configure the optional deployment step.
- Set `MODEL_VERSION` as a workflow environment variable.
- Monitor runs in the GitHub Actions tab.
- Modify `train.py` or a test, commit, and push again to trigger a new pipeline.

## Expected result
The model trains, automated tests pass, a model artifact is uploaded by GitHub Actions, and the optional deployment step either triggers (when configured) or reports that it was skipped.

Note: This is a learning demonstration using the built-in Iris dataset, not a production model or a real cloud deployment.
