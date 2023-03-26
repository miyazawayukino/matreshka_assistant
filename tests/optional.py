def Optional(value=None, default=None):
  try:
    return value
  except:
    return default

arr = {
  "message": "Hello"
}

a = Optional(arr["message2"], "Hi")

print(a)