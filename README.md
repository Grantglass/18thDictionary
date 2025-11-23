# 18th Century Dictionary Analysis

Analyze the development of English dictionaries in the 18th century using word embeddings. This project uses modern NLP techniques to study semantic change, vocabulary evolution, and definition patterns across historical dictionaries.

## Overview

The 18th century was a formative period for English lexicography, featuring landmark works like:

- **Nathan Bailey's Universal Etymological English Dictionary** (1721)
- **Samuel Johnson's A Dictionary of the English Language** (1755)
- **Thomas Sheridan's A General Dictionary of the English Language** (1780)
- **John Walker's A Critical Pronouncing Dictionary** (1791)

This toolkit enables computational analysis of how word meanings, definitions, and vocabulary evolved across these and other dictionaries.

## Features

- **Embedding Generation**: Convert dictionary definitions to semantic vectors using transformer models
- **Semantic Shift Detection**: Identify words whose meanings changed between dictionaries
- **Vocabulary Analysis**: Track words added, removed, or retained across editions
- **Neighborhood Comparison**: Analyze how semantic relationships between words evolved
- **Visualization**: Generate plots of embedding spaces, semantic shifts, and word evolution

## Installation

```bash
# Clone the repository
git clone https://github.com/Grantglass/18thDictionary.git
cd 18thDictionary

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

Dictionary data should be in JSON format. Place files in `data/dictionaries/`:

```json
{
  "name": "Johnson's Dictionary",
  "year": 1755,
  "author": "Samuel Johnson",
  "entries": [
    {
      "headword": "wit",
      "definition": "The powers of the mind; the mental faculties; the intellect",
      "part_of_speech": "n.",
      "etymology": "From Saxon wit",
      "examples": ["Great wit to madness near allied."]
    }
  ]
}
```

See `data/dictionaries/example_format.json` for a complete example.

## Usage

### Basic Analysis

```python
from src.data_loader import DictionaryLoader
from src.embeddings import EmbeddingGenerator
from src.analysis import DictionaryAnalyzer

# Load dictionaries
loader = DictionaryLoader("data/dictionaries")
dictionaries = loader.load_all()

# Generate embeddings
generator = EmbeddingGenerator()
embeddings = {
    d.name: generator.embed_dictionary(d)
    for d in dictionaries
}

# Analyze semantic shifts
analyzer = DictionaryAnalyzer(embeddings, dictionaries)
shifts = analyzer.find_semantic_shifts(
    "Bailey's Dictionary",
    "Johnson's Dictionary"
)

for shift in shifts[:10]:
    print(f"{shift.word}: {shift.shift_magnitude:.3f}")
```

### Track Word Evolution

```python
# Track how a word's meaning changed over time
evolution = analyzer.track_word_evolution("enthusiasm")

for entry in evolution:
    print(f"{entry['year']}: {entry['definition'][:80]}...")
    if entry['similarity_to_previous']:
        print(f"  Similarity to previous: {entry['similarity_to_previous']:.3f}")
```

### Visualize Results

```python
from src.visualization import DictionaryVisualizer

viz = DictionaryVisualizer()

# Plot embedding space
viz.plot_embedding_space(
    embeddings["Johnson's Dictionary"],
    highlight_words=["wit", "humour", "fancy"],
    save_path="johnson_embeddings.png"
)

# Plot semantic shifts
viz.plot_semantic_shifts(shifts, save_path="shifts.png")
```

### Run Example Script

```bash
python examples/analyze_dictionaries.py
```

## Project Structure

```
18thDictionary/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Load dictionary data
│   ├── embeddings.py     # Generate word embeddings
│   ├── analysis.py       # Semantic analysis tools
│   └── visualization.py  # Plotting and visualization
├── data/
│   └── dictionaries/     # Dictionary JSON files
├── examples/
│   └── analyze_dictionaries.py
├── notebooks/            # Jupyter notebooks
├── output/
│   ├── embeddings/       # Cached embeddings
│   └── figures/          # Generated plots
├── requirements.txt
└── README.md
```

## Data Sources

Historical dictionary texts can be obtained from:

- [Internet Archive](https://archive.org/) - Scanned historical dictionaries
- [HathiTrust Digital Library](https://www.hathitrust.org/)
- [ECCO (Eighteenth Century Collections Online)](https://www.gale.com/primary-sources/eighteenth-century-collections-online)
- [Project Gutenberg](https://www.gutenberg.org/)

Note: You may need to OCR and process scanned texts into the JSON format.

## Embedding Models

The project uses sentence-transformers for generating embeddings:

- `all-MiniLM-L6-v2` (default) - Fast, good quality
- `all-mpnet-base-v2` - More accurate, slower
- `paraphrase-multilingual-MiniLM-L12-v2` - For non-English text

## Research Applications

- **Semantic Change**: Study how word meanings evolved (e.g., "enthusiasm" shifting from religious ecstasy to general excitement)
- **Lexicographic Practice**: Compare definition styles across lexicographers
- **Vocabulary Growth**: Track the expansion of English vocabulary
- **Etymology Studies**: Analyze etymological information across sources
- **Cultural History**: Explore how definitions reflect cultural attitudes

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
