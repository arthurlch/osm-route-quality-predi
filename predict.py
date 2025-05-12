import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def prepare_model_data(edges: pd.DataFrame, narrow: pd.DataFrame):
    df = edges.copy()
    df['is_narrow'] = 0
    df.loc[narrow.index, 'is_narrow'] = 1
    feats = [c for c in ['highway', 'lanes', 'maxspeed',
                         'service', 'length'] if c in df.columns]
    return df[feats], df['is_narrow'], feats


def build_model(X, y):
    num = X.select_dtypes(include=['number']).columns.tolist()
    cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_pipe = Pipeline(
        [('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                         ('oh', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('n', num_pipe, num), ('c', cat_pipe, cat)])
    pipe = Pipeline([('pre', pre), ('clf', RandomForestClassifier(
        n_estimators=100, random_state=42))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(Xtr, ytr)
    return pipe, Xte, yte


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    print("Saved: confusion_matrix.png")

    if hasattr(model.named_steps['clf'], 'feature_importances_'):
        imp = model.named_steps['clf'].feature_importances_
        names = model.named_steps['pre'].get_feature_names_out()
        idx = np.argsort(imp)[::-1]
        plt.figure()
        plt.bar(names[idx], imp[idx])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved: feature_importance.png")
