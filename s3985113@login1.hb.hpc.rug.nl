{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "35b6a575",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import re\n",
    "import nltk\n",
    "import emoji\n",
    "#nltk.download('wordnet')\n",
    "#nltk.download('words')\n",
    "from nltk.sentiment.util import mark_negation\n",
    "from nltk.tokenize import TweetTokenizer\n",
    "from nltk.corpus import stopwords\n",
    "#from spellchecker import SpellChecker\n",
    "from nltk.stem import WordNetLemmatizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ea063ed7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# DATASET\n",
    "DATASET_COLUMNS = ['Id', 'Review', 'Sentiment']\n",
    "# Define a dictionary to map sentiment values to category names\n",
    "sentiment_labels = {1: 'Negative', 2: 'Neutral', 3: 'Positive'}\n",
    "\n",
    "\n",
    "\n",
    "# PROCESSING\n",
    "MIN_FREQ = 2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99c5941b",
   "metadata": {},
   "source": [
    "Goal of project: \n",
    "\n",
    "This notebook includes: (steps)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a50abd6b",
   "metadata": {},
   "source": [
    "### Load Dataset\n",
    "\n",
    "First, we load and explore the dataset and apply some initial processing such as setting the '*Id*' column as index and removing any empty rows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b1152d78",
   "metadata": {},
   "outputs": [],
   "source": [
    "def drop_missing(data):\n",
    "    # Remove any rows with missing values and reset the index\n",
    "    data.replace('', np.nan, inplace=True)\n",
    "    data = data.dropna()\n",
    "    data.reset_index(drop=True, inplace=True)\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a97b6f9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 2 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   Review     10000 non-null  object\n",
      " 1   Sentiment  10000 non-null  int64 \n",
      "dtypes: int64(1), object(1)\n",
      "memory usage: 156.4+ KB\n",
      "None\n",
      "\n",
      "Dataset shape: (10000, 2)\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>good and interesting</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>This class is very helpful to me. Currently, I...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>like!Prof and TAs are helpful and the discussi...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Easy to follow and includes a lot basic and im...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Really nice teacher!I could got the point eazl...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Great course - I recommend it for all, especia...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>One of the most useful course on IT Management!</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>I was disappointed because the name is mislead...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Super content. I'll definitely re-do the course</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>One of the excellent courses at Coursera for i...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0                               good and interesting          3\n",
       "1  This class is very helpful to me. Currently, I...          3\n",
       "2  like!Prof and TAs are helpful and the discussi...          3\n",
       "3  Easy to follow and includes a lot basic and im...          3\n",
       "4  Really nice teacher!I could got the point eazl...          3\n",
       "5  Great course - I recommend it for all, especia...          3\n",
       "6    One of the most useful course on IT Management!          3\n",
       "7  I was disappointed because the name is mislead...          2\n",
       "8    Super content. I'll definitely re-do the course          3\n",
       "9  One of the excellent courses at Coursera for i...          3"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load dataset\n",
    "raw_dataset_path = \"input/reviews_data.csv\"\n",
    "df_raw = pd.read_csv(raw_dataset_path)\n",
    "df_raw = df_raw[:10000]\n",
    "\n",
    "# Set ID as index\n",
    "df_raw.set_index('Id', inplace=True, drop=True)\n",
    "\n",
    "# Remove NaN rows, before cleaning text\n",
    "df_raw = drop_missing(df_raw)\n",
    "\n",
    "# Create a copy of the original DataFrame to preserve the original data\n",
    "df = df_raw.copy()\n",
    "\n",
    "print(df_raw.info())\n",
    "print(f'\\nDataset shape: {df_raw.shape}\\n')\n",
    "df_raw.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "946b0333",
   "metadata": {},
   "source": [
    "### Analysing Data (TODO)\n",
    "We then analyse the dataset by observing the distribution of review per sentiment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "94d261bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3 (Positive): 8917 reviews\n",
      "1 (Negative): 588 reviews\n",
      "2 (Neutral): 495 reviews\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Count the number of reviews per sentiment\n",
    "sentiment_counts = df['Sentiment'].value_counts()\n",
    "\n",
    "# Print the counts for each category\n",
    "for sentiment_value, count in sentiment_counts.items():\n",
    "    sentiment_name = sentiment_labels[sentiment_value]\n",
    "    print(f\"{sentiment_value} ({sentiment_name}): {count} reviews\")\n",
    "\n",
    "# Define labels and colors for the pie chart\n",
    "labels = ['Positive', 'Neutral', 'Negative']\n",
    "colors = ['limegreen', 'dodgerblue', 'red']\n",
    "\n",
    "# Plot the pie chart\n",
    "plt.pie(sentiment_counts, colors=colors, autopct='%1.1f%%',  pctdistance=0.8, textprops={'fontsize': 10, 'color': 'black'}, startangle=90)\n",
    "plt.axis('equal')  # pie as a circle\n",
    "plt.legend(labels=labels, loc='lower left')\n",
    "plt.title('Distribution of Reviews per Sentiment')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a43e7a6",
   "metadata": {},
   "source": [
    "## Clean Text"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3e3e633",
   "metadata": {},
   "source": [
    "Next, we clean the data applying the following techniques (TODO: add info):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a3edfe26",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Text Cleaning\n",
    "#spell_checker = SpellChecker()\n",
    "english_words = set(nltk.corpus.words.words())\n",
    "emojis = [\n",
    "        #HAPPY\n",
    "        \":-)\",\n",
    "        \":)\",\n",
    "        \";)\",\n",
    "        \":o)\",\n",
    "        \":]\",\n",
    "        \":3\",\n",
    "        \":c)\",\n",
    "        \":>\",\n",
    "        \"=]\",\n",
    "        \"8)\",\n",
    "        \"=)\",\n",
    "        \":}\",\n",
    "        \":^)\",\n",
    "        \":-D\",\n",
    "        \":D\",\n",
    "        \"8-D\",\n",
    "        \"8D\",\n",
    "        \"x-D\",\n",
    "        \"xD\",\n",
    "        \"X-D\",\n",
    "        \"XD\",\n",
    "        \"=-D\",\n",
    "        \"=D\",\n",
    "        \"=-3\",\n",
    "        \"=3\",\n",
    "        \":-))\",\n",
    "        \":'-)\",\n",
    "        \":')\",\n",
    "        \":*\",\n",
    "        \":^*\",\n",
    "        \">:P\",\n",
    "        \":-P\",\n",
    "        \":P\",\n",
    "        \"X-P\",\n",
    "        \"x-p\",\n",
    "        \"xp\",\n",
    "        \"XP\",\n",
    "        \":-p\",\n",
    "        \":p\",\n",
    "        \"=p\",\n",
    "        \":-b\",\n",
    "        \":b\",\n",
    "        \">:)\",\n",
    "        \">;)\",\n",
    "        \">:-)\",\n",
    "        \"<3\",\n",
    "        # SAD\n",
    "        \":L\",\n",
    "        \":-/\",\n",
    "        \">:/\",\n",
    "        \":S\",\n",
    "        \">:[\",\n",
    "        \":@\",\n",
    "        \":-(\",\n",
    "        \":[\",\n",
    "        \":-||\",\n",
    "        \"=L\",\n",
    "        \":<\",\n",
    "        \":-[\",\n",
    "        \":-<\",\n",
    "        \"=\\\\\",\n",
    "        \"=/\",\n",
    "        \">:(\",\n",
    "        \":(\",\n",
    "        \">.<\",\n",
    "        \":'-(\",\n",
    "        \":'(\",\n",
    "        \":\\\\\",\n",
    "        \":-c\",\n",
    "        \":c\",\n",
    "        \":{\",\n",
    "        \">:\\\\\",\n",
    "        \";(\",\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7d1f5731",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                                  good and interesting          3\n",
      "1     this class is very helpful to me. currently, i...          3\n",
      "2     like!prof and tas are helpful and the discussi...          3\n",
      "3     easy to follow and includes a lot basic and im...          3\n",
      "4     really nice teacher!i could got the point eazl...          3\n",
      "...                                                 ...        ...\n",
      "9995  great content and great teacher, thanks so muc...          3\n",
      "9996  great course to learn the wonders of the unive...          3\n",
      "9997  very exciting, strong structured course, the m...          3\n",
      "9998  easy to follow. anyone who is interested in as...          3\n",
      "9999    very easy to follow, and immensely interesting.          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# 1) Lowercase\n",
    "df['Review'] = df['Review'].str.lower()\n",
    "#pd.set_option('display.max_rows', df.shape[0]+1)\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "67d9fb16",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                                  good and interesting          3\n",
      "1     this class is very helpful to me. currently, i...          3\n",
      "2     like!prof and tas are helpful and the discussi...          3\n",
      "3     easy to follow and includes a lot basic and im...          3\n",
      "4     really nice teacher!i could got the point eazl...          3\n",
      "...                                                 ...        ...\n",
      "9995  great content and great teacher, thanks so muc...          3\n",
      "9996  great course to learn the wonders of the unive...          3\n",
      "9997  very exciting, strong structured course, the m...          3\n",
      "9998  easy to follow. anyone who is interested in as...          3\n",
      "9999    very easy to follow, and immensely interesting.          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# 2) Replace contractions with their standard full forms\n",
    "contraction_mapping = {\n",
    "        \"isn't\": \"is not\",\n",
    "        \"aren't\": \"are not\",\n",
    "        \"don't\": \"do not\",\n",
    "        \"doesn't\": \"does not\",\n",
    "        \"wasn't\": \"was not\",\n",
    "        \"weren't\": \"were not\",\n",
    "        \"didn't\": \"did not\",\n",
    "        \"haven't\": \"have not\",\n",
    "        \"hasn't\": \"has not\",\n",
    "        \"hadn't\": \"had not\",\n",
    "        \"won't\": \"will not\",\n",
    "        \"can't\": \"cannot\",\n",
    "        \"couldn't\": \"could not\",\n",
    "        \"shouldn't\": \"should not\",\n",
    "        \"wouldn't\": \"would not\",\n",
    "        \"mightn't\": \"might not\",\n",
    "        \"mustn't\": \"must not\",\n",
    "        }\n",
    "\n",
    "for contraction, standard in contraction_mapping.items():\n",
    "    df['Review'] = df['Review'].str.replace(contraction, standard)\n",
    "\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8492481d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                                  good and interesting          3\n",
      "1     this class is very helpful to me. currently, i...          3\n",
      "2     like prof and tas are helpful and the discussi...          3\n",
      "3     easy to follow and includes a lot basic and im...          3\n",
      "4     really nice teacher i could got the point eazl...          3\n",
      "...                                                 ...        ...\n",
      "9995  great content and great teacher, thanks so muc...          3\n",
      "9996  great course to learn the wonders of the unive...          3\n",
      "9997  very exciting, strong structured course, the m...          3\n",
      "9998  easy to follow. anyone who is interested in as...          3\n",
      "9999    very easy to follow, and immensely interesting.          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ferga\\AppData\\Local\\Temp\\ipykernel_25096\\603766481.py:4: FutureWarning: The default value of regex will change from True to False in a future version.\n",
      "  df['Review'] = df['Review'].str.replace(pattern, ' ')\n"
     ]
    }
   ],
   "source": [
    "# 3) Remove punctuation in between words e.g. \"course.sometimes\" \n",
    "# and replace with space\n",
    "pattern = r'(?<=\\w)[^\\w\\s]+(?=\\w)'\n",
    "df['Review'] = df['Review'].str.replace(pattern, ' ')\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "39da71ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>[good, and, interesting]</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>[this, class, is, very, helpful, to, me, ., cu...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>[like, prof, and, tas, are, helpful, and, the,...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>[easy, to, follow, and, includes, a, lot, basi...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>[really, nice, teacher, i, could, got, the, po...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0                           [good, and, interesting]          3\n",
       "1  [this, class, is, very, helpful, to, me, ., cu...          3\n",
       "2  [like, prof, and, tas, are, helpful, and, the,...          3\n",
       "3  [easy, to, follow, and, includes, a, lot, basi...          3\n",
       "4  [really, nice, teacher, i, could, got, the, po...          3"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 4) Tokenize text into individual words (removes all extra spaces \\s)\n",
    "tokenizer = TweetTokenizer()\n",
    "df['Review'] = df['Review'].apply(tokenizer.tokenize)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "796988cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                              [good, and, interesting]          3\n",
      "1     [this, class, is, very, helpful, to, me, ., cu...          3\n",
      "2     [like, prof, and, tas, are, helpful, and, the,...          3\n",
      "3     [easy, to, follow, and, includes, a, lot, basi...          3\n",
      "4     [really, nice, teacher, i, could, got, the, po...          3\n",
      "...                                                 ...        ...\n",
      "9995  [great, content, and, great, teacher, thanks, ...          3\n",
      "9996  [great, course, to, learn, the, wonders, of, t...          3\n",
      "9997  [very, exciting, strong, structured, course, t...          3\n",
      "9998  [easy, to, follow, ., anyone, who, is, interes...          3\n",
      "9999  [very, easy, to, follow, and, immensely, inter...          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# TODO: three dots i.e. ... not removed\n",
    "# 5) Remove punctuation first in between words (typo),\n",
    "# and then all punctuation and numerals except for tokenized emojis\n",
    "pattern = r\"[^\\w\\s\" + \"\".join(re.escape(e) for e in emojis + list(emoji.EMOJI_DATA.keys())) + \"]|[\\d]+\" # match non-emoji special characters\n",
    "df['Review'] = df['Review'].apply(lambda tokens: [token for token in tokens if not re.match(pattern, token)])\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "18e56982",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                              [good, and, interesting]          3\n",
      "1     [this, class, is, very, helpful, to, me, curre...          3\n",
      "2     [like, prof, and, tas, are, helpful, and, the,...          3\n",
      "3     [easy, to, follow, and, includes, lot, basic, ...          3\n",
      "4     [really, nice, teacher, could, got, the, point...          3\n",
      "...                                                 ...        ...\n",
      "9995  [great, content, and, great, teacher, thanks, ...          3\n",
      "9996  [great, course, to, learn, the, wonders, of, t...          3\n",
      "9997  [very, exciting, strong, structured, course, t...          3\n",
      "9998  [easy, to, follow, anyone, who, is, interested...          3\n",
      "9999  [very, easy, to, follow, and, immensely, inter...          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# 6) Remove single characters\n",
    "df['Review'] = df['Review'].apply(lambda tokens: [word for word in tokens if len(word) > 1])\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c9b4d712",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n# 7) Correct Spelling\\ncorrected_tokens = []\\nfor token in filtered_tokens:\\n    if token in emojis or token in emoji.EMOJI_DATA.keys():\\n        corrected_tokens.append(token)  # If token is an emoji, add it to the corrected tokens\\n    else:\\n        corrected_token = spell_checker.correction(token)\\n        if corrected_token is not None:\\n            corrected_tokens.append(corrected_token)\\n#print('spell-check: '+str(corrected_tokens))\\n\""
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# TODO: package not loading\n",
    "'''\n",
    "# 7) Correct Spelling\n",
    "corrected_tokens = []\n",
    "for token in filtered_tokens:\n",
    "    if token in emojis or token in emoji.EMOJI_DATA.keys():\n",
    "        corrected_tokens.append(token)  # If token is an emoji, add it to the corrected tokens\n",
    "    else:\n",
    "        corrected_token = spell_checker.correction(token)\n",
    "        if corrected_token is not None:\n",
    "            corrected_tokens.append(corrected_token)\n",
    "#print('spell-check: '+str(corrected_tokens))\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "81593c1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: note to self (to be added to word-doc): If you check token by token, it also removes english words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5a8964b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                              [good, and, interesting]          3\n",
      "1     [this, class, is, very, helpful, to, me, curre...          3\n",
      "2     [like, prof, and, tas, are, helpful, and, the,...          3\n",
      "3     [easy, to, follow, and, includes, lot, basic, ...          3\n",
      "4     [really, nice, teacher, could, got, the, point...          3\n",
      "...                                                 ...        ...\n",
      "9995  [great, content, and, great, teacher, thanks, ...          3\n",
      "9996  [great, course, to, learn, the, wonders, of, t...          3\n",
      "9997  [very, exciting, strong, structured, course, t...          3\n",
      "9998  [easy, to, follow, anyone, who, is, interested...          3\n",
      "9999  [very, easy, to, follow, and, immensely, inter...          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# 8) Perform negation tagging\n",
    "df['Review'] = df['Review'].apply(mark_negation)\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6d9cf56d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                                   [good, interesting]          3\n",
      "1     [class, helpful, currently, still, learning, c...          3\n",
      "2     [like, prof, tas, helpful, discussion, among, ...          3\n",
      "3     [easy, follow, includes, lot, basic, important...          3\n",
      "4     [really, nice, teacher, could, got, point, eaz...          3\n",
      "...                                                 ...        ...\n",
      "9995  [great, content, great, teacher, thanks, much,...          3\n",
      "9996  [great, course, learn, wonders, universe, unde...          3\n",
      "9997  [exciting, strong, structured, course, difficu...          3\n",
      "9998  [easy, follow, anyone, interested, astronomy, ...          3\n",
      "9999             [easy, follow, immensely, interesting]          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# 9) Remove stopwords --> also removes words like 'not'\n",
    "stop_words = set(stopwords.words('english'))\n",
    "df['Review'] = df['Review'].apply(lambda tokens: [token for token in tokens if token not in stop_words])\n",
    "df['Review'] = df['Review'].apply(lambda tokens: [token for token in tokens if token.split('_')[0] not in stop_words])\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "09cebc1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                 Review  Sentiment\n",
      "0                                   [good, interesting]          3\n",
      "1     [class, helpful, currently, still, learning, c...          3\n",
      "2     [like, prof, ta, helpful, discussion, among, s...          3\n",
      "3     [easy, follow, includes, lot, basic, important...          3\n",
      "4     [really, nice, teacher, could, got, point, eaz...          3\n",
      "...                                                 ...        ...\n",
      "9995  [great, content, great, teacher, thanks, much,...          3\n",
      "9996  [great, course, learn, wonder, universe, under...          3\n",
      "9997  [exciting, strong, structured, course, difficu...          3\n",
      "9998  [easy, follow, anyone, interested, astronomy, ...          3\n",
      "9999             [easy, follow, immensely, interesting]          3\n",
      "\n",
      "[10000 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "# TODO: also lemmatize word removing _NEG\n",
    "# 10) Lemmatize words using WordNetLemmatizer\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "df['Review'] = df['Review'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e25a06a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>good interesting</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>class helpful currently still learning class m...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>like prof ta helpful discussion among student ...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>easy follow includes lot basic important techn...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>really nice teacher could got point eazliy</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0                                   good interesting          3\n",
       "1  class helpful currently still learning class m...          3\n",
       "2  like prof ta helpful discussion among student ...          3\n",
       "3  easy follow includes lot basic important techn...          3\n",
       "4         really nice teacher could got point eazliy          3"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Convert preprocessed tokens back to string\n",
    "df['Review'] = df['Review'].apply(' '.join)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "25434a1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape before: (10000, 2)\n",
      "Shape after preprocessing, before removing empty rows: (10000, 2)\n",
      "Shape after preprocessing, after removing empty rows: (9996, 2)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(f'Shape before: {df_raw.shape}')\n",
    "print(f'Shape after preprocessing, before removing empty rows: {df.shape}')\n",
    "\n",
    "# Remove NaN rows, after cleaning text\n",
    "df = drop_missing(df) \n",
    "print(f'Shape after preprocessing, after removing empty rows: {df.shape}\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "b120f716",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_dataset_path = \"cleaned_input/cleaned_data.csv\"\n",
    "df.to_csv(cleaned_dataset_path, sep=',', index_label='Id')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "df0708cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: create word clouds"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa29eef3",
   "metadata": {},
   "source": [
    "# Preprocess Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "42534f64",
   "metadata": {},
   "outputs": [],
   "source": [
    "import multiprocessing\n",
    "import os.path\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from numpy import asarray\n",
    "import gensim.downloader as api\n",
    "from collections import Counter\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from keras.utils import pad_sequences"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f65f114",
   "metadata": {},
   "source": [
    "### Split train and test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "73f09fce",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data Distribution:\n",
      "* train: 6396\n",
      "* validation: 1600\n",
      "* test: 2000\n",
      "\n",
      "x_train: 9465    interesting course done_NEG calculus_NEG years...\n",
      "4793    interesting course focus social political aspe...\n",
      "938                             enriching content coupled\n",
      "9461                 started looking real interesting far\n",
      "4142    great loved fact could watch video follow step...\n",
      "Name: Review, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Split dataset into training and test sets\n",
    "x_train, x_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'],\n",
    "                                                    test_size=0.2, random_state=42)\n",
    "# Split the training dataset further into training and validation sets\n",
    "x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)\n",
    "print(\"Data Distribution:\\n* train: {}\\n* validation: {}\\n* test: {}\\n\".format(len(x_train), len(x_val), len(x_test)))\n",
    "\n",
    "print(\"x_train: {}\".format(x_train.head()))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ceb8e8d3",
   "metadata": {},
   "source": [
    "### Create vocabulary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "25b30607",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('course', 4490), ('great', 1348), ('good', 1147), ('really', 673), ('course_NEG', 637), ('well', 590), ('lot', 587), ('interesting', 548), ('excellent', 544), ('much', 418), ('thank', 415), ('useful', 372), ('learn', 359), ('would', 348), ('easy', 344), ('understand', 339), ('professor', 335), ('way', 330), ('time', 329), ('lecture', 326), ('one', 322), ('material', 321), ('like', 320), ('video', 307), ('thanks', 303), ('content', 301), ('class', 299), ('information', 294), ('teacher', 291), ('data', 291)]\n"
     ]
    }
   ],
   "source": [
    "# Count words to create vocabulary\n",
    "word_counter = Counter()\n",
    "for review in x_train:\n",
    "    word_counter.update(review.split())\n",
    "\n",
    "print(word_counter.most_common(30))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "0aa7a58b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vocabulary size of 10552 reduced to 5305.\n",
      "\n",
      "Vocabulary (first 50 tokens):\n",
      "['interesting', 'course', 'done_NEG', 'calculus_NEG', 'years_NEG', 'wonderful_NEG', 'get_NEG', 'back_NEG', 'really_NEG', 'enjoyed_NEG', 'jim_NEG', 'enthusiasm_NEG', 'explanation_NEG', 'fundamentals_NEG', '.', '._NEG', 'would_NEG', 'helpful_NEG', 'errors_NEG', 'marking_NEG', 'quizzes_NEG', 'corrected_NEG', 'pointed_NEG', 'months_NEG', 'ago_NEG', 'focus', 'social', 'political', 'aspect', 'well', 'known', 'historical', 'information', 'provided', 'different', 'country', 'around', 'world', 'highlighted', 'various', 'president', 'dealt', 'epidemic', 'amazing', 'big', 'influence', 'disease', 'would', 'preferred', 'bio']\n"
     ]
    }
   ],
   "source": [
    "# Filter vocabulary by removing words with frequency less than a set minimum frequency\n",
    "vocab = [word for word, count in word_counter.items() if count >= MIN_FREQ]\n",
    "vocab_size = len(vocab)\n",
    "print(\"Vocabulary size of {} reduced to {}.\\n\".format(len(word_counter), vocab_size))\n",
    "print(\"Vocabulary (first 50 tokens):\\n{}\".format(vocab[:50]))\n",
    "\n",
    "# Save filtered vocabulary to a txt file TODO: pickle?\n",
    "vocab_file = 'processed/vocab.txt'\n",
    "with open(vocab_file, 'w') as f:\n",
    "    f.write('\\n'.join(vocab))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83fc4550",
   "metadata": {},
   "source": [
    "### Filter data with vocabulary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "31b1e29b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def freq_filter_dataset(docs, filename, vocab):\n",
    "    filtered_dataset = []\n",
    "    for doc in docs:\n",
    "        filtered_text = ' '.join([word for word in doc.split() if word in vocab])\n",
    "        filtered_dataset.append(filtered_text)\n",
    "\n",
    "    # Save filtered dataset to a txt file\n",
    "    filtered_filename = f'processed/filtered_{str(filename)}.txt'\n",
    "    with open(filtered_filename, 'w') as f:\n",
    "        f.write('\\n'.join(filtered_dataset))\n",
    "\n",
    "    # Convert the processed documents back to pandas.Series\n",
    "    filtered_dataset = pd.Series(filtered_dataset, index=docs.index)\n",
    "\n",
    "    # Convert empty rows to '<empty>'\n",
    "    placeholder = \"<empty>\"\n",
    "    filtered_dataset.replace('', placeholder, inplace=True)\n",
    "    \n",
    "    # Count the number of rows with '<empty>'\n",
    "    num_empty_rows = filtered_dataset.str.count('<empty>').sum()\n",
    "    print(f'Number of rows with <empty> for {filename}: {num_empty_rows}')\n",
    "\n",
    "    # Save filled dataset to a txt file\n",
    "    filled_filename = f'processed/filled_{str(filename)}.txt'\n",
    "    with open(filled_filename, 'w') as f:\n",
    "        f.write('\\n'.join(filtered_dataset))\n",
    "    \n",
    "    return filtered_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e136714f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of rows with <empty> for x_train: 7\n",
      "Number of rows with <empty> for x_val: 3\n",
      "Number of rows with <empty> for x_test: 1\n",
      "\n",
      "Data Distribution:\n",
      "* train: 6396\n",
      "* validation: 1600\n",
      "* test: 2000\n",
      "\n",
      "x_train - updated:\n",
      "9465    interesting course done_NEG calculus_NEG years...\n",
      "4793    interesting course focus social political aspe...\n",
      "938                             enriching content coupled\n",
      "9461                 started looking real interesting far\n",
      "4142    great loved fact could watch video follow step...\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Filter dataset based on vocabulary\n",
    "x_train = freq_filter_dataset(x_train, \"x_train\", vocab)\n",
    "x_val = freq_filter_dataset(x_val, \"x_val\", vocab)\n",
    "x_test = freq_filter_dataset(x_test, \"x_test\", vocab)\n",
    "\n",
    "print(\"\\nData Distribution:\\n* train: {}\\n* validation: {}\\n* test: {}\\n\".format(len(x_train), len(x_val), len(x_test)))\n",
    "print(\"x_train - updated:\")\n",
    "print(x_train.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8c6d509",
   "metadata": {},
   "source": [
    "## TF-IDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a9f624e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# PARAMS \n",
    "\n",
    "# TF-IDF\n",
    "MAX_FEATURES = 10000\n",
    "MAX_DF = 0.95\n",
    "MIN_DF = 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "4ccee997",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize TfidfVectorizer with the filtered vocabulary\n",
    "vectorizer = TfidfVectorizer(\n",
    "    max_features=MAX_FEATURES, # maximum number of features to keep, check unique vocabs and determine based on that, high causes saprse metrics and low value causes loss in important words/vocab\n",
    "    vocabulary=vocab,\n",
    "    lowercase=False,\n",
    "    ngram_range=(1, 1),  # range of n-grams, only unigrams now\n",
    "    max_df=MAX_DF,  # ignore terms that have a document frequency strictly higher than the threshold\n",
    "    min_df=MIN_DF,  # ignore terms that have a document frequency strictly lower than the threshold.\n",
    "    use_idf=True,  # enable IDF weighting\n",
    "    smooth_idf=True,  # smooth IDF weights --> provides stability, reduces run time errors\n",
    "    sublinear_tf=True  # apply sublinear scaling to term frequencies\n",
    ")\n",
    "\n",
    "# Fit and transform the training set\n",
    "x_train_tfidf = vectorizer.fit_transform(x_train)\n",
    "\n",
    "# Transform the validation and testing set\n",
    "x_val_tfidf = vectorizer.transform(x_val)\n",
    "x_test_tfidf = vectorizer.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "40e25575",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_tfidf_data(data, data_name, feature_names):\n",
    "    # Save the matrix with feature names as a DataFrame\n",
    "    data = pd.DataFrame(data.toarray(), columns=feature_names)\n",
    "    data.to_csv(f'processed/{data_name}.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "40d62af4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get feature names\n",
    "feature_names = vectorizer.get_feature_names_out()\n",
    "\n",
    "# Save vectorized data\n",
    "save_tfidf_data(x_train_tfidf, \"train_tfidf\", feature_names)\n",
    "save_tfidf_data(x_train_tfidf, \"val_tfidf\", feature_names)\n",
    "save_tfidf_data(x_test_tfidf, \"test_tfidf\", feature_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "daad3bbf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Given vocabulary-size : 5305,\n",
      "\n",
      "Data Shape:\n",
      "* train: (6396, 5305)\n",
      "* validation: (1600, 5305)\n",
      "* test: (2000, 5305)\n",
      "\n",
      "x_train_tfidf:\n",
      "  (0, 24)\t0.222595054069153\n",
      "  (0, 23)\t0.24267421407303155\n",
      "  (0, 22)\t0.2510078184286611\n",
      "  (0, 21)\t0.24267421407303155\n",
      "  (0, 20)\t0.1725607370417893\n",
      "  (0, 19)\t0.2510078184286611\n",
      "  (0, 18)\t0.22646320541311804\n",
      "  (0, 17)\t0.18154533743556625\n",
      "  (0, 16)\t0.14538084404291787\n",
      "  (0, 13)\t0.2510078184286611\n",
      "  (0, 12)\t0.20438545269391686\n",
      "  (0, 11)\t0.23621016834604439\n",
      "  (0, 10)\t0.21613100834216586\n",
      "  (0, 9)\t0.19077033841702548\n",
      "  (0, 8)\t0.14614656539760387\n",
      "  (0, 7)\t0.20251589406527448\n",
      "  (0, 6)\t0.15001471674156888\n",
      "  (0, 5)\t0.222595054069153\n",
      "  (0, 4)\t0.21084949842090403\n",
      "  (0, 3)\t0.32300226062525333\n",
      "  (0, 2)\t0.1798408396783738\n",
      "  (0, 1)\t0.04556545911289766\n",
      "  (0, 0)\t0.10095566896569971\n",
      "  (1, 61)\t0.1589808655755073\n",
      "  (1, 60)\t0.14607716544780977\n",
      "  :\t:\n",
      "  (6393, 215)\t0.2960348794114244\n",
      "  (6393, 203)\t0.14117342675667546\n",
      "  (6393, 143)\t0.10866791302379811\n",
      "  (6393, 128)\t0.10911215829486592\n",
      "  (6393, 123)\t0.1087561765743259\n",
      "  (6393, 117)\t0.16536046130267182\n",
      "  (6393, 106)\t0.17119905925442805\n",
      "  (6393, 85)\t0.13097417754703983\n",
      "  (6393, 79)\t0.07511190616982731\n",
      "  (6393, 69)\t0.07095086402133934\n",
      "  (6393, 55)\t0.13264145479420703\n",
      "  (6393, 32)\t0.1112559396925171\n",
      "  (6393, 16)\t0.22837581933930443\n",
      "  (6393, 1)\t0.07157785557640324\n",
      "  (6394, 604)\t0.5167593192779814\n",
      "  (6394, 603)\t0.43261448870735064\n",
      "  (6394, 369)\t0.3486815218230552\n",
      "  (6394, 365)\t0.35172120281055197\n",
      "  (6394, 304)\t0.2822876046534488\n",
      "  (6394, 265)\t0.28147653418875235\n",
      "  (6394, 79)\t0.1847358699498358\n",
      "  (6394, 44)\t0.3278341489937793\n",
      "  (6395, 1370)\t0.7316405243067894\n",
      "  (6395, 128)\t0.48062850306286414\n",
      "  (6395, 92)\t0.4834236084798057\n"
     ]
    }
   ],
   "source": [
    "print(\"Given vocabulary-size : {},\".format(vocab_size))\n",
    "print(\"\\nData Shape:\\n* train: {}\\n* validation: {}\\n* test: {}\\n\".format(x_train_tfidf.shape, x_val_tfidf.shape, x_test_tfidf.shape))\n",
    "print(\"x_train_tfidf:\\n{}\".format(x_train_tfidf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "199b0d3d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Data Types:\n",
      "x_train_tfidf - type: <class 'scipy.sparse._csr.csr_matrix'>\n",
      "x_val_tfidf - type: <class 'scipy.sparse._csr.csr_matrix'>\n",
      "y-train - type: <class 'pandas.core.series.Series'>\n"
     ]
    }
   ],
   "source": [
    "print(f'\\nData Types:\\nx_train_tfidf - type: {type(x_train_tfidf)}\\nx_val_tfidf - type: {type(x_val_tfidf)}\\ny-train - type: {type(y_train)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ca1595c",
   "metadata": {},
   "source": [
    "# Classical ML"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "03df5a06",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f4939b93",
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_metrics(y_true, y_pred, model_name):\n",
    "    # Calculate metrics\n",
    "    accuracy = accuracy_score(y_true, y_pred)\n",
    "    # TODO: not anymore?? Handle the zero-division error when there are no predicted samples for a label\n",
    "    # only interested in labels that were predicted at least once\n",
    "    precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))\n",
    "    recall = recall_score(y_true, y_pred, average='weighted')\n",
    "    f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))\n",
    "    \n",
    "    # Calculate classification report\n",
    "    report = classification_report(y_true, y_pred)\n",
    "    \n",
    "    # Print results\n",
    "    print(f'{model_name} Accuracy: {accuracy}')\n",
    "    print(f'{model_name} Precision: {precision}')\n",
    "    print(f'{model_name} Recall: {recall}')\n",
    "    print(f'{model_name} F1 Score: {f1_score}')\n",
    "    print()\n",
    "    print(report)\n",
    "\n",
    "    # Save results\n",
    "    save_dir = f'results/{model_name}_results.txt'\n",
    "    with open(save_dir, 'w') as file:\n",
    "        file.write(f'{model_name} Accuracy: {accuracy}\\n')\n",
    "        file.write(f'{model_name} Precision: {precision}\\n')\n",
    "        file.write(f'{model_name} Recall: {recall}\\n')\n",
    "        file.write(f'{model_name} F1 Score: {f1_score}\\n')\n",
    "        file.write(\"\\n\\n\")\n",
    "        file.write(report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "db2c35bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_confusion_matrix(y_true, y_pred, labels, model_name):\n",
    "    save_dir = f'results/{model_name}_confusion_matrix.png'\n",
    "    \n",
    "    cnf_matrix = confusion_matrix(y_true, y_pred, labels=labels)\n",
    "    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=labels)\n",
    "    cm_display.plot()\n",
    "    plt.show()\n",
    "    plt.savefig(save)\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6ab685ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot(history, save_dir, model_name):\n",
    "    accuracy_plot = f'{save_dir}/{model_name}_plot.png'\n",
    "    loss_plot = f'{save_dir}/{model_name}_loss_plot.png'\n",
    "    \n",
    "    accuracy = history.history['accuracy']\n",
    "    #val_accuracy = history.history['val_accuracy']\n",
    "\n",
    "    epochs = range(len(accuracy))\n",
    "    plt.plot(epochs, accuracy, 'r', label='Training acc')\n",
    "    #plt.plot(epochs, val_accuracy, 'b', label='Validation acc')\n",
    "\n",
    "    plt.title(f'{model_name} Model Accuracy')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "    plt.savefig(save_dir)\n",
    "    plt.close()\n",
    "    \n",
    "    loss = history.history['loss']\n",
    "    #val_loss = history.history['val_loss']\n",
    "\n",
    "    epochs = range(len(loss))\n",
    "    plt.plot(epochs, loss, 'r', label='Training acc')\n",
    "    #plt.plot(epochs, val_loss, 'b', label='Validation acc')\n",
    "\n",
    "    plt.title(f'{model_name} Model Loss')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "    plt.savefig(save_dir)\n",
    "    plt.close()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "7aefe79a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate_model(model, model_name, x_test, y_test):\n",
    "    y_pred = model.predict(x_test)\n",
    "    print(f'{model_name} Testing complete!')\n",
    "\n",
    "    # Calculate and save metrics\n",
    "    calculate_metrics(y_test, y_pred, model_name)\n",
    "\n",
    "    # Plot accuracy\n",
    "    #plot(history, model_name)\n",
    "\n",
    "    # Plot confusion matrix\n",
    "    senti_labels = ['negative', 'neutral', 'positive']\n",
    "    plot_confusion_matrix(y_test, y_pred, senti_labels, model_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b616c18",
   "metadata": {},
   "source": [
    "## 1. Random Forest"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6e69ba1",
   "metadata": {},
   "source": [
    "### Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "1c04b22b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the parameter grid for grid search\n",
    "rf_param_grid = {\n",
    "    'n_estimators': [100, 200, 300],\n",
    "    'max_depth': [None, 5, 10],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node\n",
    "    'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "681b15f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create instances of the Random Forest model & fit on training data\n",
    "rf_classifier = RandomForestClassifier()\n",
    "#rf_classifier.fit(x_train_tfidf, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3057a5f1",
   "metadata": {},
   "source": [
    "### Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "54c2eefb",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_25096\\2087145558.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Perform grid search\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mgrid_search\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mGridSearchCV\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mestimator\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mrf_classifier\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mparam_grid\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mrf_param_grid\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcv\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mgrid_search\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train_tfidf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;31m# Get the best parameters and best score from grid search\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, groups, **fit_params)\u001b[0m\n\u001b[0;32m    889\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mresults\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    890\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 891\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_run_search\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mevaluate_cands\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    892\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    893\u001b[0m             \u001b[1;31m# multimetric is determined here because in the case of a callable\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py\u001b[0m in \u001b[0;36m_run_search\u001b[1;34m(self, evaluate_cands)\u001b[0m\n\u001b[0;32m   1390\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_run_search\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mevaluate_cands\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1391\u001b[0m         \u001b[1;34m\"\"\"Search all cands in param_grid\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1392\u001b[1;33m         \u001b[0mevaluate_cands\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mParameterGrid\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mparam_grid\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1393\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1394\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py\u001b[0m in \u001b[0;36mevaluate_cands\u001b[1;34m(cand_params, cv, more_results)\u001b[0m\n\u001b[0;32m    836\u001b[0m                     )\n\u001b[0;32m    837\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 838\u001b[1;33m                 out = parallel(\n\u001b[0m\u001b[0;32m    839\u001b[0m                     delayed(_fit_and_score)(\n\u001b[0;32m    840\u001b[0m                         \u001b[0mclone\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbase_estimator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1044\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_original_iterator\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1045\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1046\u001b[1;33m             \u001b[1;32mwhile\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdispatch_one_batch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1047\u001b[0m                 \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1048\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36mdispatch_one_batch\u001b[1;34m(self, iterator)\u001b[0m\n\u001b[0;32m    859\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mFalse\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    860\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 861\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_dispatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtasks\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    862\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    863\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m_dispatch\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    777\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    778\u001b[0m             \u001b[0mjob_idx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 779\u001b[1;33m             \u001b[0mjob\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcb\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    780\u001b[0m             \u001b[1;31m# A job can complete so quickly than its callback is\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    781\u001b[0m             \u001b[1;31m# called before we get here, causing self._jobs to\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36mapply_async\u001b[1;34m(self, func, callback)\u001b[0m\n\u001b[0;32m    206\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    207\u001b[0m         \u001b[1;34m\"\"\"Schedule a func to be run\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 208\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mImmediateResult\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    209\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    210\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    570\u001b[0m         \u001b[1;31m# Don't delay the application, to avoid keeping the input\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    571\u001b[0m         \u001b[1;31m# arguments in memory\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 572\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresults\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    573\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    574\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    260\u001b[0m         \u001b[1;31m# change the default number of processes to -1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    261\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 262\u001b[1;33m             return [func(*args, **kwargs)\n\u001b[0m\u001b[0;32m    263\u001b[0m                     for func, args, kwargs in self.items]\n\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    260\u001b[0m         \u001b[1;31m# change the default number of processes to -1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    261\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 262\u001b[1;33m             return [func(*args, **kwargs)\n\u001b[0m\u001b[0;32m    263\u001b[0m                     for func, args, kwargs in self.items]\n\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\fixes.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    214\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__call__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    215\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mconfig_context\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconfig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 216\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfunction\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    217\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    218\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py\u001b[0m in \u001b[0;36m_fit_and_score\u001b[1;34m(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, cand_progress, error_score)\u001b[0m\n\u001b[0;32m    678\u001b[0m             \u001b[0mestimator\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mfit_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    679\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 680\u001b[1;33m             \u001b[0mestimator\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mfit_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    681\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    682\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    448\u001b[0m             \u001b[1;31m# parallel_backend contexts set at a higher level,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    449\u001b[0m             \u001b[1;31m# since correctness does not rely on using threads.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 450\u001b[1;33m             trees = Parallel(\n\u001b[0m\u001b[0;32m    451\u001b[0m                 \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn_jobs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    452\u001b[0m                 \u001b[0mverbose\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mverbose\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1044\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_original_iterator\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1045\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1046\u001b[1;33m             \u001b[1;32mwhile\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdispatch_one_batch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1047\u001b[0m                 \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1048\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36mdispatch_one_batch\u001b[1;34m(self, iterator)\u001b[0m\n\u001b[0;32m    859\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mFalse\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    860\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 861\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_dispatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtasks\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    862\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    863\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m_dispatch\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    777\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    778\u001b[0m             \u001b[0mjob_idx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 779\u001b[1;33m             \u001b[0mjob\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcb\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    780\u001b[0m             \u001b[1;31m# A job can complete so quickly than its callback is\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    781\u001b[0m             \u001b[1;31m# called before we get here, causing self._jobs to\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36mapply_async\u001b[1;34m(self, func, callback)\u001b[0m\n\u001b[0;32m    206\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    207\u001b[0m         \u001b[1;34m\"\"\"Schedule a func to be run\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 208\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mImmediateResult\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    209\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    210\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    570\u001b[0m         \u001b[1;31m# Don't delay the application, to avoid keeping the input\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    571\u001b[0m         \u001b[1;31m# arguments in memory\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 572\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresults\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    573\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    574\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    260\u001b[0m         \u001b[1;31m# change the default number of processes to -1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    261\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 262\u001b[1;33m             return [func(*args, **kwargs)\n\u001b[0m\u001b[0;32m    263\u001b[0m                     for func, args, kwargs in self.items]\n\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    260\u001b[0m         \u001b[1;31m# change the default number of processes to -1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    261\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 262\u001b[1;33m             return [func(*args, **kwargs)\n\u001b[0m\u001b[0;32m    263\u001b[0m                     for func, args, kwargs in self.items]\n\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\fixes.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    214\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__call__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    215\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mconfig_context\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconfig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 216\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfunction\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    217\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    218\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\u001b[0m in \u001b[0;36m_parallel_build_trees\u001b[1;34m(tree, forest, X, y, sample_weight, tree_idx, n_trees, verbose, class_weight, n_samples_bootstrap)\u001b[0m\n\u001b[0;32m    183\u001b[0m             \u001b[0mcurr_sample_weight\u001b[0m \u001b[1;33m*=\u001b[0m \u001b[0mcompute_sample_weight\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"balanced\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindices\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mindices\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    184\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 185\u001b[1;33m         \u001b[0mtree\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcurr_sample_weight\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcheck_input\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    186\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    187\u001b[0m         \u001b[0mtree\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msample_weight\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcheck_input\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    935\u001b[0m         \"\"\"\n\u001b[0;32m    936\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 937\u001b[1;33m         super().fit(\n\u001b[0m\u001b[0;32m    938\u001b[0m             \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    939\u001b[0m             \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    418\u001b[0m             )\n\u001b[0;32m    419\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 420\u001b[1;33m         \u001b[0mbuilder\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbuild\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtree_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    421\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    422\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn_outputs_\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mis_classifier\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# Perform grid search\n",
    "grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, cv=5)\n",
    "grid_search.fit(x_train_tfidf, y_train.values)\n",
    "\n",
    "# Get the best parameters and best score from grid search\n",
    "best_params = grid_search.best_params_\n",
    "best_score = grid_search.best_score_\n",
    "print(\"Best Parameters: \", best_params)\n",
    "print(\"Best Score: \", best_score)\n",
    "\n",
    "# Get the mean test scores and standard deviations of test scores for all parameter combinations\n",
    "results_df = pd.DataFrame(grid_search.cv_results_)\n",
    "results_df = results_df.sort_values(by='mean_test_score', ascending=False)\n",
    "\n",
    "# Print the top 3 combinations of mean test score and standard deviation of test scores\n",
    "print(\"\\nTop 3 Combinations:\")\n",
    "for i in range(3):\n",
    "    params = results_df.iloc[i]['params']\n",
    "    mean_score = results_df.iloc[i]['mean_test_score']\n",
    "    std_score = results_df.iloc[i]['std_test_score']\n",
    "    print(f\"Combination {i+1}: {params} | mean score: {mean_score} +- (std score: {std_score})\")\n",
    "\n",
    "# Use the best model to make predictions on the validation set\n",
    "best_model = grid_search.best_estimator_\n",
    "y_pred = best_model.predict(x_val_tfidf)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c75503a1",
   "metadata": {},
   "source": [
    "### Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bf69266",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the models on the validation data\n",
    "evaluate_model(rf_classifier,\"RF\", x_val_tfidf, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98a1e81b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the models on the test data\n",
    "evaluate_model(rf_classifier, \"RF\", x_test_tfidf, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1790120",
   "metadata": {},
   "source": [
    "### 2. Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9acda108",
   "metadata": {},
   "outputs": [],
   "source": [
    "# PARAMS HERE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21458d8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create instances of the Naive Bayes model & fit on training data\n",
    "nb_model = MultinomialNB()\n",
    "nb_model.fit(x_train_tfidf, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3287aee",
   "metadata": {},
   "source": [
    "### 3. SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5e05435",
   "metadata": {},
   "outputs": [],
   "source": [
    "# PARAMS HERE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccbbc59b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create instances of the SVM model & fit on training data\n",
    "svm_model = SVC()\n",
    "svm_model.fit(x_train_tfidf, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ceb654f8",
   "metadata": {},
   "source": [
    "## Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94f4da28",
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(nb_model, \"NB\", x_val_tfidf, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb6b755f",
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(svm_model, \"SVM\", x_val_tfidf, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17434f7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(nb_model, \"NB\", x_test_tfidf, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "470a9be5",
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(svm_model, \"SVM\", x_test_tfidf, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9982ae27",
   "metadata": {},
   "source": [
    "# Encode Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5d99acb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find maximum sequence length\n",
    "max_seq_length = max([len(doc.split()) for doc in x_train_filtered])\n",
    "print(f'\\nMax doc length: {max_seq_length}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b895c28",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit tokenizer (on training data)\n",
    "tokenizer = Tokenizer()\n",
    "# Remove default filters, including punctuation\n",
    "tokenizer.filters = \"\"  \n",
    "# Disable lowercase conversion\n",
    "tokenizer.lower = False  \n",
    "tokenizer.fit_on_texts(x_train_filtered) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbb76b88",
   "metadata": {},
   "outputs": [],
   "source": [
    "def encode_text(lines, tokenizer, max_length):\n",
    "    # Integer encode\n",
    "    encoded_seq = tokenizer.texts_to_sequences(lines)\n",
    "    # Pad the encoded sequences\n",
    "    padded = pad_sequences(encoded_seq, maxlen=max_length, padding='post')\n",
    "\n",
    "    # Save padded sequences to a text file\n",
    "    # TODO: note, only encodes final input sent\n",
    "    encoded_file = \"processed/encoded.txt\"\n",
    "    with open(encoded_file, 'w') as file:\n",
    "        for sequence in padded:\n",
    "            line = \" \".join(str(num) for num in sequence)\n",
    "            file.write(line + \"\\n\")\n",
    "\n",
    "    return padded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7655e1b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encode Data\n",
    "x_train_encoded = encode_text(x_train_filtered, tokenizer, max_seq_length)\n",
    "x_val_encoded = encode_text(x_val_filtered, tokenizer, max_seq_length)\n",
    "x_test_encoded = encode_text(x_test_filtered, tokenizer, max_seq_length)\n",
    "\n",
    "print(f'x_train_encoded - shape: {x_train_encoded.shape}')\n",
    "print(f'x_val_encoded - shape: {x_val_encoded.shape}')\n",
    "print(f'x_test_encoded - shape: {x_test_encoded.shape}')\n",
    "print(x_val_encoded[:3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e193568e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Restructure labels\n",
    "y_train = y_train.values\n",
    "y_val = y_val.values\n",
    "y_test = y_test.values\n",
    "print(f'y_train - shape: {y_train.shape}')\n",
    "print(f'y_val - shape: {y_val.shape}')\n",
    "print(f'y_test- shape: {y_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "527d5290",
   "metadata": {},
   "source": [
    "# Word2Vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72d81df0",
   "metadata": {},
   "outputs": [],
   "source": [
    "for word in tokenizer.word_index.keys():\n",
    "    print(word)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80ad4aec",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Total vocabulary size plus 0 for unknown words\n",
    "embedding_vocab_size = len(tokenizer.word_index) + 1\n",
    "print(f'embedding_vocab_size: {embedding_vocab_size}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b891123",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_embedding():\n",
    "    # Check if the pre-trained Word2Vec model is already downloaded\n",
    "    #w2v_pretrained_model = \"glove-twitter-100\"\n",
    "    w2v_pretrained_model = \"glove-wiki-gigaword-100\"\n",
    "    w2v_pretrained_model_filename = \"./data/raw/\" + str(w2v_pretrained_model) + \"-word2vec.txt\"\n",
    "    if not os.path.exists(w2v_pretrained_model_filename):\n",
    "        print(\"\\nw2v model doesn't exist\")\n",
    "        # If the model does not exist, download it\n",
    "        model = api.load(\"glove-twitter-100\")\n",
    "        # Save the word2vec embeddings in the appropriate format\n",
    "        model.save_word2vec_format(w2v_pretrained_model_filename, binary=False)\n",
    "\n",
    "    # load embedding into memory, skip first line\n",
    "    print(\"Loading w2v model...\")\n",
    "    file = open(w2v_pretrained_model_filename, 'r', encoding='utf8')\n",
    "    lines = file.readlines()[1:]\n",
    "    file.close()\n",
    "    # create a map of words to vectors\n",
    "    embedding = dict()\n",
    "    for line in lines:\n",
    "        parts = line.split()\n",
    "        # key is string word, value is numpy array for vector\n",
    "        embedding[parts[0]] = asarray(parts[1:], dtype='float32')\n",
    "    return embedding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2010e9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_embedding = load_embedding()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f8aec82",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_weight_matrix(embedding, tokenizer):\n",
    "    # create a weight matrix for the Embedding layer from a loaded embedding\n",
    "\n",
    "    # define weight matrix dimensions with all 0\n",
    "    weight_matrix = np.zeros((embedding_vocab_size, w2v_dim_size))\n",
    "    # step vocab, store vectors using the Tokenizer's integer mapping\n",
    "    count_all = 0\n",
    "    count_na = 0\n",
    "    for word, i in tokenizer.word_index.items():\n",
    "        # TODO: important note, pretrained word2vec model removes all neg_ and emojis (also other words) that are\n",
    "        #  not defined in the model it These values should prob? also be removed from the vocab (and update vocab size) to avoid mismatch in the embedding layer\n",
    "        if word in embedding.keys():\n",
    "            # print(embedding.get(word)[:3])\n",
    "            weight_matrix[i] = embedding.get(word)\n",
    "        else:\n",
    "            #print(word)\n",
    "            count_na += 1\n",
    "        count_all += 1\n",
    "    print(f'count_na/count_all: {str(count_na)}/{count_all}')\n",
    "    print(f\"embedding matrix shape: {weight_matrix.shape}\")\n",
    "\n",
    "    # save model in ASCII (word2vec) format\n",
    "    weight_matrix_filename = str(data_dir) + 'weight_matrix_word2vec.txt'\n",
    "    with open(weight_matrix_filename, 'w') as f:\n",
    "        f.write('\\n'.join(' '.join(str(x) for x in row) for row in weight_matrix))\n",
    "    print(\"Saving weight embedding matrix to a txt file...\")\n",
    "\n",
    "    return weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3a6844c",
   "metadata": {},
   "outputs": [],
   "source": [
    "w2v_embedding_vectors = get_weight_matrix(raw_embedding, tokenizer)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
