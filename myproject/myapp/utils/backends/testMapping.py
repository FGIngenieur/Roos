from quoteMapping import *

mapper = SemanticMapping()
mapper.train([
    ("Le chat dort.", 1.0),
    ("La voiture est cassée.", 10.0)
])

mapper.save("my_mapping")

loaded = SemanticMapping.load("my_mapping")

print(loaded.predict("Le félin se repose."))
