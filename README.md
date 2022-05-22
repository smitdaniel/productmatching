# productmatching

This short program allows to match products added during a particular time window with users, based on their history
of interest.

Note that the source data is not gitted. To run the program, you need to place it to `data/source.xlsx`.

## Processing

The data processing takes place in `data_reader.RawData`, which performs basic reading, cleaning, and some 
type transformation (multi-value strings into tuples etc.).

Distance matrix between countries appearing in the data is managed by `data_reader.Distance` class and is cached
in `data/distance.json` to avoid OSM calls on each run.

## Scoring logic

The scoring of customer preferences itself, as well as additional evaluation of suggestions feasibility (costs,
distance, etc.) are performed in `matchmaker.MatchMaker` class. The class also stores the results into 
`out/results.xlsx`.

## Documentation

Documentation explaining the method and giving a brief overview of implementation and validation is available in
`doc/documentation.pdf`.

## Implementation

The file `definitions.py` contains important program paths, `config/config.yaml` contains configuration of the 
program run, which is also commented there. The dependencies are listed in `doc/requirements.txt`.
