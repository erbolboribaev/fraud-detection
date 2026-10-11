"""Spark schema of the scored-transaction event, derived from the API-side
contract in src/events.py so the producer and the lake can't silently diverge."""
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.events import EVENT_FIELDS

_TYPES = {
    "string": StringType(),
    "double": DoubleType(),
    "int": IntegerType(),
    "boolean": BooleanType(),
}

SCORED_EVENT_SCHEMA = StructType(
    [StructField(name, _TYPES[kind], nullable=True) for name, kind in EVENT_FIELDS.items()]
)
