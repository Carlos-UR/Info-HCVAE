import argparse
import json

def load_json(file):
  with open(file, "r", encoding='utf-8') as reader:
        return json.load(reader)

def write_text(file, text):
  with open(file, "w", encoding='utf-8') as writer:
        writer.write(text)

def main(args):
  input_data = load_json(args.input_file)

  exceptions = ['instrucciones_aplicacion', 'condiciones_aplicacion']

  producto = ''

  # Entradas del JSON
  for entry in input_data:
    # Claves de cada entrada
    for key in entry.keys():
      if(entry[key]):
        if (key in exceptions):
          subkey = entry[key]
          # Subclaves de cada clave
          for k in subkey.keys():
            if(subkey[k]):
              value = subkey[k].replace(f'\n', '')
              producto += f"{k} : {value}."
              #print(k + " : " + str(subkey[k]))
        else:
          value = entry[key].replace(f'\n', '')
          producto += f"{key} : {value}."
          #print(key + " : " + str(entry[key]));
    producto += f'\n'

  write_text(args.output_file, producto)
  print(f"{args.input_file} transformado correctamente a {args.output_file}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", required=True, type=str)
  parser.add_argument("--output_file", type=str, default='./out.txt')

  args = parser.parse_args()

  main(args)