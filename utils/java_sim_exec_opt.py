import subprocess
import os
from difflib import SequenceMatcher

def execute_java_code(java_code):
    class_name = "Test"
    java_filename = f"{class_name}.java"
    class_filename = f"{class_name}.class"

    # Wrap the provided Java code in a class structure
    full_java_code = f"public class {class_name} {{\n{java_code}\n}}"

    # Write the complete Java code to a file
    with open(java_filename, "w") as file:
        file.write(full_java_code)

    try:
        # Compile the Java file
        subprocess.run(["javac", java_filename], check=True, stderr=subprocess.PIPE)
        # Execute the compiled class
        result = subprocess.run(["java", class_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation/execution: {e.stderr.decode()}")
        return None
    finally:
        # Clean up
        os.remove(java_filename)
        if os.path.exists(class_filename):
            os.remove(class_filename)

    return result.stdout.strip()

def similarity(java_code1, java_code2):

    try:
        output1 = execute_java_code(java_code1)
        output2 = execute_java_code(java_code2)
        
        if output1 is None or output2 is None:
            print("Error executing one or both of the Java code snippets.")
            return
        
        # Compute similarity ratio
        similarity_ratio = SequenceMatcher(None, output1, output2).ratio()
        return similarity_ratio
    except Exception as e:
        return 0

