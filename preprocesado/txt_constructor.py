import argparse
import json
import pdb

def load_json(file):
  with open(file, "r", encoding='utf-8') as reader:
        return json.load(reader)

def write_text(file, text):
  with open(file, "w", encoding='utf-8') as writer:
        writer.write(text)

def main(args):
  input_data = load_json(args.input_file)

  exceptions = ['instrucciones_aplicacion', 'condiciones_aplicacion']

  all_product_text = ''
  if (args.limit):
    input_data = input_data[0:args.limit]
  # Entradas del JSON
  for entry in input_data:
    product_text = ''
    nombre = entry["nombre"]
    tipo = entry["tipo"]
    descripcion = entry["descripcion"]
    ambito_aplicacion = entry["ambito_aplicacion"]
    ip = entry["instrucciones_aplicacion"]
    ca = entry["condiciones_aplicacion"]
    obra_nueva = entry["obra_nueva"]
    restauracion = entry["restauracion"]
    preparacion_soporte = entry["preparacion_soporte"]

    product_text += f"{nombre} es {descripcion}" if descripcion else f"{nombre} " 
    product_text += f"de tipo {tipo}." if tipo else ""
    product_text += f"El ámbito de aplicación de {nombre} es {ambito_aplicacion}." if ambito_aplicacion else ""
    product_text += f"Las instrucciones de aplicación son: "
    
    if (ip):
      product_text += f"El método de aplicación de {nombre} es {ip['metodo_aplicacion']}." if ip['metodo_aplicacion'] else ""
      product_text += f"{ip['dilucion']}." if ip["dilucion"] else ""
      product_text += f"{ip['limpieza_herramientas']}." if ip["limpieza_herramientas"] else ""
      product_text += f"{ip['tiempo_secado']}." if ip["tiempo_secado"] else ""
      product_text += f"{ip['advertencias']}." if ip["advertencias"] else ""
    
    if (ca):
      product_text += f"Las condiciones de aplicación son: "
      product_text += f"{ca['humedad_substrato']}." if ca["humedad_substrato"] else ""
      product_text += f"El punto de rocío de {nombre} es {ca['punto_rocio']}." if ca["punto_rocio"] else ""
    
    product_text += f"{obra_nueva}." if obra_nueva else ""
    product_text += f"Restauración : {restauracion}." if restauracion else ""
    product_text += f"{preparacion_soporte}." if preparacion_soporte else ""
    product_text = product_text.replace(f'\n', '')
    all_product_text += f"{product_text}\n"
  write_text(args.output_file, all_product_text)
  print(f"TEXTO CREADO: {args.input_file} transformado correctamente a {args.output_file}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", required=True, type=str, help="File where load the input")
  parser.add_argument("--output_file", type=str, default='./out.txt', help="File name where store the output" )
  parser.add_argument('--limit', type=int, help="Number of product to read.")

  args = parser.parse_args()

  main(args)