import lancedb as ldb
import polars as pl
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

model = get_registry().get("huggingface").create(name='facebook/bart-base')

class TextModel(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


df = pl.DataFrame({"text": ["hi", "hello", "sayonara", "goodbye world", "adios"]})
db = ldb.connect("~/.lancedb-bart")
table = db.create_table("greets", schema=TextModel, mode="overwrite")
table.add(df)


query = "How to say goodbye in Japanese?"
actual = table.search(query).limit(1).to_pydantic(TextModel)[0]
print(actual.text)



