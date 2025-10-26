This the backend file for the project.

Order of execution is:
1. Running preprocess.py
2. Running train_ncf.py (When running the project for the first time, this will load the initial epochs, from which you can continue training using train_ncf_contd.py)
3. To train further epochs, load from last checkpoint till your desired checkpoint through train_ncf_cont.py (You can also implement early stopping, I haven't)
4. Load your desired epoch using model_loader.py
5. Test your model locally using run_reccomender.py (Optional, recommended to test if model works or not)
6. Run Flask Backend using app.py
