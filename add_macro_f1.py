import os
import re

def calculate_macro_f1(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the classification report section
    report_match = re.search(r'--- CLASSIFICATION REPORT ---(.*)', content, re.DOTALL)
    if not report_match:
        # Try finding it without the dashed header if the format varies slightly, 
        # but based on the provided file, it has the header.
        return None

    report_text = report_match.group(1)
    
    f1_scores = []
    
    # iterating over lines in the report
    for line in report_text.split('\n'):
        line = line.strip()
        if not line: continue
        
        parts = line.split()
        # specific check for lines starting with a number (the class label)
        # The structure is: class_name precision recall f1-score support
        if len(parts) >= 4 and parts[0].isdigit():
             try:
                 # parts[3] corresponds to f1-score
                 f1 = float(parts[3])
                 f1_scores.append(f1)
             except ValueError:
                 continue
    
    if not f1_scores:
        return None
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1

def update_file(file_path, macro_f1):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    inserted = False
    
    for line in lines:
        # Check if already exists just to be safe (though we check before calling)
        if "MACRO F1-SCORE:" in line:
            # Skip existing line if we are re-running/updating
            continue
            
        new_lines.append(line)
        
        # Insert after "F1 SCORE: ..."
        if "F1 SCORE:" in line and not inserted:
            new_lines.append(f"MACRO F1-SCORE: {macro_f1:.4f}\n")
            inserted = True
            
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def process_directory(root_dir):
    print(f"Scanning directory: {root_dir}")
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "report_metrics.txt":
                file_path = os.path.join(root, file)
                
                # Check if already processed (simple check)
                # We can choose to overwrite or skip. The user said "adds", implying it's not there.
                # But if I messed up, I might want to overwrite. 
                # I'll implement a read-check to see if I should skip or validly update.
                
                needs_update = True
                with open(file_path, 'r') as f:
                    if "MACRO F1-SCORE" in f.read():
                        needs_update = False
                
                # However, if I want to FORCE update to ensure it's calculated correctly, I should ignore the check.
                # But let's stick to adding if missing for safety.
                # User asked to "add".
                
                if not needs_update:
                    # Let's verify if we want to overwrite. Maybe it was there from a previous run?
                    # The prompt implies doing it now. I'll stick to skipping if present to avoid duplicating lines endlessly if script is run multiple times.
                    # Actually, my update_file logic removes existing MACRO F1-SCORE line if present while rewriting, so it's safe to always run.
                    pass 

                macro_f1 = calculate_macro_f1(file_path)
                if macro_f1 is not None:
                    update_file(file_path, macro_f1)
                    print(f"Updated {file_path} -> MACRO F1: {macro_f1:.4f}")
                    count += 1
                else:
                    print(f"Could not calculate Macro F1 for {file_path}")
    print(f"Total files updated: {count}")

if __name__ == "__main__":
    # Start from current directory
    process_directory(".")
