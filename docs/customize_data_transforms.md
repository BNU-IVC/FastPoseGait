## A New Data Transform
If you want to write a new data transform, you can write a new class in `data/transform.py`

e.g, NewTranform in `data/transform.py`:
```python
class NewTranform():
    def __init__(self):
        pass
    def __call__(self, data):
        return data

```