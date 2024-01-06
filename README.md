first train the data sets
using the data
make this changes during training datasets


from sklearn.decomposition import PCA

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense, Dropout, GaussianNoise, Conv1D

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

import seaborn as sns

---



sns.barplot(x='Acedamic percentage in Operating Systems', y='Suggested Job Role', data=df)

plt.xlim(5, 60)

plt.show()

---



add  models folder
model.sav
model.h5

scaleX

files in models folder


run the cmd in terminal 

python -m venv .venv

.venv/Scripts/activate

pip install -r requirements.txt

python app.py
