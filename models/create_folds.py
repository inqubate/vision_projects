import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
    df = pd.read_csv("../input/labeledTrainData.tsv", sep="\t")
    df.loc[:,"kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    targets = df.sentiment.values

    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (trn, val) in enumerate(skf.split(X=df, y=targets)):
        df.loc[val, "kfold"] = fold

    df.to_csv("../input/train_folds.csv", index=False)



