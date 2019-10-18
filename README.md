# Skip-Thought-Vectors
Implementation of the research paper paper skip thought vectors.

Skip thought vectors uses a RNN based model to predict the context of surrounding sentences.
As compared to previous studies which use word vector embeddings to predict context, this paper converts the word embeddings to sentence embeddings and uses Long Short Term Memory Units or Gated Recurrent Units(GRUs) to process the sentences.

The model was trained on random stories scraped from the web from the BookCorpus dataset.
The trained model was then used for classification tasks on the movie reviews dataset and Q-A dataset.
