# /patent_analysis/divitree_plotter.py

import os
import re
import ast
import pandas as pd
import numpy as np

from IPython.display import display, HTML, clear_output

import plotly.graph_objects as go
import plotly.offline as pyo
from ipywidgets import widgets, HBox, Output
from plotly.graph_objs import FigureWidget
from plotly.subplots import make_subplots

# Initialize Plotly offline mode
pyo.init_notebook_mode(connected=True)

# Map interleaving_type --> color (for this whole plot OR per row if type column exists)
COLOR_MAP = {
    "Priorities":       "#8fbbda",  # blue
    "Applications":     "#ffbf86",  # orange
    "Parents":          "#96d096",  # green
    "Publications":     "#eb9393",  # red/pink
    "Citations":        "#cab3de",  # purple/mauve
    "Classifications":  "#c6aba5",  # taupe/neutral
    "Parties":          "#f1bbe0",  # pink/lilac
    "Legal Events":     "#bfbfbf",  # light gray
    "Procedural Codes": "#7ec3a7",  # teal
    "Images":           "#dede90",  # yellow-green
    "Bibliographic Register Codes": "#ffde59",  # bright yellow
    "Events Register Codes":        "#ff7f50",  # coral
    "Procedural Steps":             "#6495ED",  # cornflower blue
    "UPP Codes":                    "#20b2aa"   # light sea green
}

def detect_indent_and_type(line, known_applications, interleaving_type=None):
    # print("known_applications:", known_applications)
    # print("5. interleaving_type:", interleaving_type)
    line = line.rstrip('\n')
    if line.strip() == "":
        return None, None, None, None

    prefix_match = re.match(r'^([., ]+)', line)
    indent = prefix_match.group(1) if prefix_match else ''
    indent_level = indent.count(',') + indent.count('.')
    content = line[len(indent):].strip()

    if indent.startswith(','):
        entry_type = "Application"
        line_type = "comma"
        known_applications.add(content)
    elif indent.startswith('.'):
        entry_type = "Application" if content in known_applications else interleaving_type or "No selection"
        line_type = "dot"
    else:
        entry_type = "no further selection"
        line_type = "unknown"

    # print("entry_type:", entry_type)
    return indent_level, content, entry_type, line_type

def parse_item_list(item_str):
    try:
        items = ast.literal_eval(item_str)
        if not isinstance(items, list):
            items = [items]
    except:
        items = str(item_str).split(',')
    return [str(i).strip().strip("'\"") for i in items if str(i).strip().upper() not in ['NONE', 'NULL', '']]

class DiviTreePlotter:
    def __init__(self, workdir="./output"):
        self.workdir = workdir
        self.image_width = 700  # pixels
        self.persistent_color_map = {}  # maps node id → color
        self.persistent_type_map = {}   # maps node id → type
        if not hasattr(self, "last_selected_fixed_type"):
            self.last_selected_fixed_type = None
        # print("Recorded last_selected_fixed_type:", self.last_selected_fixed_type)
    
    def assign_branch_colors(self, df, fromWhere=None):
        """
        Apply COLOR_MAP for all node types,
        and refine 'Applications' with distinct per-app colours,
        while preserving fixed colours like UPP Codes.
        """
        # print("🚨 assign_branch_colors ENTERED fromWhere=", fromWhere, flush=True)
        
        # Ensure 'color' and 'hover_bg' columns exist
        for col in ['color', 'hover_bg']:
            if col not in df.columns:
                df[col] = None
                
        if "type" in df.columns:
            df["type"] = df["type"].fillna("").astype(str).str.strip()

        # 🔄 Identify which fixed types (from COLOR_MAP) exist in this data
        present_fixed_types = [t for t in COLOR_MAP.keys() 
                               if t in df["type"].values and t != 'Application']
        # print("Present COLOR_MAP types in this df (excluding 'Application'):", present_fixed_types)

        # Record the user's first-run selection if available and not already recorded
        prev_type = self.last_selected_fixed_type
        if present_fixed_types: # and self.last_selected_fixed_type is None:
            # prefer to store the explicit present fixed type the user chose in their first run
            self.last_selected_fixed_type = present_fixed_types[0]
            # if self.last_selected_fixed_type != prev_type:
            #     print("Updated last_selected_fixed_type to:", self.last_selected_fixed_type)
            # else:
            #     print("Recorded last_selected_fixed_type:", self.last_selected_fixed_type)
        
        # 1️⃣ Replace "No selection" only if there are fixed types to remap to
        if "No selection" in df["type"].values:
            candidates = present_fixed_types + ([self.last_selected_fixed_type] if self.last_selected_fixed_type else []) \
                         + list(reversed(self.persistent_type_map.values())) \
                         + [t for c in self.persistent_color_map.values() for t, col in COLOR_MAP.items() if col == c]
            target_type = next((t for t in candidates if t in COLOR_MAP), None)
            if target_type:
                df["type"] = df["type"].replace({"No selection": target_type})
                print(f"Mapped 'No selection' to {target_type} (persistent-aware)")
            else:
                print("No valid target_type found for 'No selection' — leaving as-is for now")
            
        # 2️⃣ Start with fixed colour map (may assign NaN for Applications if not in COLOR_MAP)
        df["color"] = df["type"].map(COLOR_MAP)
        df["hover_bg"] = df["color"].copy()
        
        # 3️⃣ Assign colors for static types
        #    (covers any fixed type present, not only the small subset)
        for typ in df["type"].unique():
            if typ in COLOR_MAP:
                mask = df["type"] == typ
                df.loc[mask, ["color", "hover_bg"]] = COLOR_MAP[typ]
                # store persistent mapping using a stable key derived from each node id
                for raw_nid in df.loc[mask, "id"]:
                    stable_nid = str(raw_nid).split('[')[0].strip()
                    self.persistent_color_map[stable_nid] = COLOR_MAP[typ]
                    self.persistent_type_map[stable_nid] = typ
                # also remember this typ as last_selected_fixed_type for future No selection mapping
                if self.last_selected_fixed_type is None:
                    self.last_selected_fixed_type = typ

        # 4️⃣ Handle Application nodes, 🔑 Applications: restore or save
        if "id" in df.columns:
            # Extract stable IDs
            df['stable_id'] = df['id'].astype(str).str.split('[').str[0].str.strip()
    
            # Step 1: Restore persistent colors where available
            persistent_colors = df['stable_id'].map(self.persistent_color_map)
            df['color'] = persistent_colors.combine_first(df.get('color'))  # keep existing colors if persistent missing
            df['hover_bg'] = df['color']
    
            # Record types for persistent_type_map if not already present
            missing_types_mask = df['stable_id'].isin(self.persistent_color_map) & \
                                 ~df['stable_id'].isin(self.persistent_type_map) & \
                                 df['type'].notna()
            for sid, typ in zip(df.loc[missing_types_mask, 'stable_id'], df.loc[missing_types_mask, 'type']):
                self.persistent_type_map[sid] = typ

            # Step 2: Assign colors from COLOR_MAP where color is still missing
            missing_color_mask = df['color'].isna() & df['type'].isin(COLOR_MAP)
            df.loc[missing_color_mask, 'color'] = df.loc[missing_color_mask, 'type'].map(COLOR_MAP)
            df.loc[missing_color_mask, 'hover_bg'] = df.loc[missing_color_mask, 'color']

            # Step 3: Update persistent maps for newly assigned colors
            for sid, col, typ in zip(df.loc[missing_color_mask, 'stable_id'], 
                                     df.loc[missing_color_mask, 'color'], 
                                     df.loc[missing_color_mask, 'type']):
                self.persistent_color_map[sid] = col
                self.persistent_type_map[sid] = typ

            # Step 4: Log remaining nodes with no color            
            mask_log = df['color'].isna() & (df['type'] != 'Application')
            for sid, node_type, node_id_raw in zip(df.loc[mask_log, 'stable_id'], 
                                                   df.loc[mask_log, 'type'], 
                                                   df.loc[mask_log, 'id']):
                print(f"⚠️ Skipping node {node_id_raw} of type '{node_type}': no color available")

            # Optional: drop helper column
            df.drop(columns='stable_id', inplace=True)

        # 5️⃣ Warn about truly missing types
        missing_types = set(df["type"].unique()) - set(COLOR_MAP.keys()) - {"Application"}
        if missing_types:
            print(f"Warning: missing color for {missing_types}")

        # print("Unique types in df after filtering:", df["type"].unique())
        # print("Sample color assignments:", df[["type", "color", "hover_bg"]].drop_duplicates().head(10))
    
        return df
        
    def find_latest_output_files(self):
        directory = self.workdir
        # print("directory:", directory)
        first_files = [f for f in os.listdir(directory) if f.endswith("_first_output.txt")]
        # print("first_files:", first_files)
        second_files = [f for f in os.listdir(directory) if f.endswith("_second_output.txt")]
        # print("second_files:", second_files)
        
        first_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        second_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        
        first_path = os.path.join(directory, first_files[0]) if first_files else ""
        second_path = os.path.join(directory, second_files[0]) if second_files else ""
        # print("first_path:", first_path)    
        # print("second_path:", second_path)
        
        return (first_path, second_path)  # Tuple, no None
    
    def read_tree_data(self, file_path, interleaving_type=None, reference_mode=None):
        root_display_name = os.path.basename(file_path).split("_")[0]
        root_node = {
                     'display_name': root_display_name, 
                     'type': 'Application',
                     'children': []
                    }
        current_level = {0: root_node}
        known_applications = {root_display_name}
    
        # Map comment lines to interleaving types
        comment_to_type = {
            "Tree displays application data": "Applications",
            "Tree displays priority data": "Priorities",
            "Tree displays parent data": "Parents",
            "Tree displays publication data": "Publications",
            "Tree displays citation data": "Citations",
            "Tree displays classification data": "Classifications",
            "Tree displays parties data": "Parties",
            "Tree includes images": "Images",
            "Tree includes legal events": "Legal Events",
            "Tree includes procedural codes": "Procedural Codes",
            "Tree includes bibliographic register codes": "Bibliographic Register Codes",
            "Tree includes events register codes": "Events Register Codes",
            "Tree includes procedural steps": "Procedural Steps",
            "Tree includes unitary patent protection (UPP) codes": "UPP Codes"          
        }
            
        seen_rows = set()  # To avoid duplicate entries
        
        # print("read_tree_data before instruction to check: interleaving_type, reference_mode:", interleaving_type, reference_mode)
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.rstrip()

                # Only scan header/comment lines (not tree data lines) for type detection.
                # Tree data lines start with ',' or '.' — skip type detection for those.
                is_data_line = line.startswith((',', '.'))

                if not is_data_line:
                    # Check for reference mode pattern in header lines only
                    if 'in the reference mode publication' in line.lower():
                        reference_mode = 'publication'
                    elif 'in the reference mode application' in line.lower():
                        reference_mode = 'application'

                    # Detect interleaving_type from header comment lines only,
                    # and only if not already supplied by the caller or detected earlier.
                    if interleaving_type is None:
                        for key, value in comment_to_type.items():
                            if key.lower() in line.lower():
                                interleaving_type = value
                                # print(f"🔎 Found interleaving_type: {interleaving_type} from line: {line}")
                                break

                if not line or not is_data_line:
                    continue
                
                result = detect_indent_and_type(line, known_applications, interleaving_type)
                if not result:
                    continue
                    
                indent_level, display_name, entry_type, line_type = result
                # print("1. interleaving_type:", interleaving_type)
                # print("indent_level:", indent_level)            
                # print("display_name:", display_name)
                # print("entry_type:", entry_type)

                # ✅ SKIP root-level lines entirely (no node, no DataFrame row)
                if indent_level == 0:
                    continue
                
                parent_level = max(l for l in current_level if l <= indent_level)
                parent_node = current_level[parent_level]
            
                if parent_node is None or display_name == parent_node['display_name']:
                    # print(f"Error: No parent found at indent level {indent_level} for line: {line}")
                    continue
                
                row_path = f"{parent_node['display_name']} > {display_name}"
                if row_path in seen_rows:
                    continue
                seen_rows.add(row_path)
            
                node = {
                    'display_name': display_name,
                    'type': entry_type,
                    'line_type': line_type,
                    'depth': indent_level,
                    'children': []
                }
            
                parent_node['children'].append(node)
                current_level[indent_level + 1] = node
                
        if interleaving_type is None:
            interleaving_type = "no further selection"
            
        # print("read_tree_data as detected: interleaving_type, reference_mode:", interleaving_type, reference_mode)
        return {'Root': root_node}, interleaving_type, reference_mode
        
    def _extract_date_from_label(self, label: str):
        """
        Extracts the first valid date from a label string like:
          "RFEE: Renewal fee payment; 03 [20040526]"
          "GRAA: (Expected) grant [2012-04-27]"
            "0009199INVT: Change - inventor [🟠20040319]"
        Returns pd.Timestamp or NaT, with a date (YYYY-MM-DD or YYYYMMDD) from procedural/legal labels.
        """
        if not isinstance(label, str):
            return pd.NaT
                
        # Match YYYY-MM-DD or YYYYMMDD, with or without emojis/brackets
        patterns = [
            r"\[🟠?(\d{4}-\d{2}-\d{2})\]",   # e.g. [🟠2012-04-27]
            r"\[🟠?(\d{8})\]",               # e.g. [🟠20040526]
            r"(\d{4}-\d{2}-\d{2})",          # plain YYYY-MM-DD
            r"(\d{8})"                       # plain YYYYMMDD
        ]

        for pat in patterns:
            m = re.search(pat, label)
            if m:
                val = m.group(1)
                try:
                    return pd.to_datetime(val, errors='coerce')
                except Exception:
                    continue
        return pd.NaT
        
    # @staticmethod
    def create_grouped_items_df(self, data, parent_key=''):
        ids, parents, values, types, depths, line_types, event_dates = [], [], [], [], [], [], []

        def compute_node_value(node):
            """Compute value as sum of all descendant leaves."""
            tooltip_types = {
                "Parents", "Priorities", "Publications", "Citations", "Classifications",
                "Parties", "Legal Events", "Procedural Codes", "Bibliographic Register Codes",
                "Events Register Codes", "Procedural Steps", "UPP Codes"
            }
            if node.get('type') in tooltip_types:
                try:
                    items = parse_item_list(node.get('display_name', ''))
                    return len([c for c in items if str(c).strip().upper() != 'NONE'])
                except Exception:
                    return 1
            if not node.get('children'):
                return 1
            return sum(compute_node_value(c) for c in node.get('children', []))

        for key, node in data.items():
            if not isinstance(node, dict):
                continue

            display_name = node.get('display_name', key)
            node_type = node.get('type', 'Unknown')
            node_value = compute_node_value(node)
            date_val = self._extract_date_from_label(display_name)

            # --- Special case: root node ---
            if parent_key == '' and key == 'Root':
                for child in node.get('children', []):
                    child_df = self.create_grouped_items_df({child['display_name']: child}, parent_key='')
                    for col in ['id', 'parent', 'value', 'type', 'depth', 'line_type', 'event_date']:
                        locals()[col + 's'].extend(child_df[col])
                continue

            tooltip_types = {
                "Parents", "Priorities", "Publications", "Citations", "Classifications",
                "Parties", "Legal Events", "Procedural Codes", "Bibliographic Register Codes",
                "Events Register Codes", "Procedural Steps", "UPP Codes"
            }

            # --- Add current node ---
            if node_type in tooltip_types:
                try:
                    items = parse_item_list(display_name)
                    items = [c for c in items if str(c).strip().upper() != 'NONE']
                except Exception:
                    items = [] if str(display_name).strip().upper() == 'NONE' else [display_name]
                if items:
                    ids.append(f"{parent_key}: {items}")
                    parents.append(parent_key)
                    values.append(len(items))
                    types.append(node_type)
                    depths.append(node.get('depth', 0))
                    line_types.append(node.get('line_type'))
                    event_dates.append(date_val)
            else:
                if display_name != parent_key:
                    ids.append(display_name)
                    parents.append(parent_key)
                    values.append(node_value)
                    types.append(node_type)
                    depths.append(node.get('depth', 0))
                    line_types.append(node.get('line_type'))
                    event_dates.append(date_val)

            # --- Recurse into children ---
            for child in node.get('children', []):
                child_df = self.create_grouped_items_df({child['display_name']: child}, parent_key=display_name)
                for col in ['id', 'parent', 'value', 'type', 'depth', 'line_type', 'event_date']:
                    locals()[col + 's'].extend(child_df[col])

        # --- Sanity check: all lists must have equal length ---
        n = len(ids)
        assert all(len(lst) == n for lst in [parents, values, types, depths, line_types, event_dates]), (
            f"Length mismatch: ids={len(ids)}, parents={len(parents)}, values={len(values)}, "
            f"types={len(types)}, depths={len(depths)}, line_types={len(line_types)}, event_dates={len(event_dates)}"
        )

        # --- Construct DataFrame ---
        df = pd.DataFrame({
            'id': ids,
            'parent': parents,
            'value': values,
            'type': types,
            'depth': depths,
            'line_type': line_types,
            'event_date': event_dates
        })

        # --- Extract event_date from id text if needed ---
        df['event_date'] = df['id'].apply(self._extract_date_from_label).fillna(df['event_date'])
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

        # --- Propagate missing event dates downwards ---
        id_to_date = dict(zip(df['id'], df['event_date']))
        for idx, row in df.iterrows():
            if pd.isna(row['event_date']) and row['parent'] in id_to_date:
                df.at[idx, 'event_date'] = id_to_date[row['parent']]

        # Fill remaining missing with earliest
        earliest = df['event_date'].dropna().min()
        if pd.notna(earliest):
            df['event_date'] = df['event_date'].fillna(earliest)

        # # --- Debug summary ---
        # valid_dates = df['event_date'].dropna()
        # print(f"🧭 Extracted {len(valid_dates)} valid event dates "
        #       f"({valid_dates.min() if len(valid_dates)>0 else 'None'} → {valid_dates.max() if len(valid_dates)>0 else 'None'})")

        return df

    # Improved tooltip formatter for both citations and classifications
    def build_tooltip(self, row, df=None): # def build_tooltip(self, row, interleaving_type, df=None):
        """
        Generate a formatted tooltip for a sunburst node.
        Handles ANSI codes, lists, images, and general formatting.
        """
        # Regex pattern to catch all ANSI escape sequences like \x1b[38;5;214m or \x1b[0m
        ANSI_ESCAPE_RE = re.compile(r'\x1B(?:\[[0-?]*[ -/]*[@-~])')

        def remove_ansi(text: str) -> str:
            """Remove ANSI escape codes from a string."""
            return ANSI_ESCAPE_RE.sub('', text)
    
        node_id = row['id']
        value = row['value']
        app_id = row.get('app_id', '')  # ✅ ensures we can prepend parent app
        pub_id = row.get('pub_id', '')  # publication/patent propagated
        parent = row['parent']
        depth = row.get('depth', 0)
        line_type = row.get('line_type', 'unknown')
        
        interleaving_type = row.get('interleaving_type', 'no further selection')
        reference_mode = row.get('reference_mode', 'Application')
    
        if not value or str(value).strip().upper() == "NONE":
           return ""
        
        text = str(node_id).strip()
        prefix = text
        
        # � For dotted application lines, show parent publication instead
        if interleaving_type == "Applications" and line_type == "dot" and parent:
            # pub_id may be empty string; fallback to prefix
            prefix = str(parent).strip()
        elif interleaving_type == "Parents" and line_type == "dot" and node_id:
            prefix = str(node_id).strip()
            
        # print("prefix:", prefix)
        
        # Split out any leading "prefix: [list...]" pattern
        if ": [" in text:
            prefix, remainder = text.split(": [", 1)
            prefix = prefix.strip()
            text = "[" + remainder.strip().rstrip("]") + "]"  # ensure list string is well-formed    

        if text.startswith("[") and text.endswith("]"):
            try:
                items = ast.literal_eval(text)
                if not isinstance(items, list):
                    items = [str(items)]
            except Exception:
                # Remove brackets and keep as single-item list
                items = [text[1:-1].strip()]
        else:
            # Single string without brackets, keep as one-item list
            items = [text]

        # print("items:", items)
                
        # � Remove ANSI escape codes from everything
        items = [remove_ansi(str(i)).strip() for i in items if str(i).strip()]
        # print("items:", items)
        
        # General nodes (Applications, Publications, etc.)
        bullet_items = []
        designation_buffer = []
        for item in items:
            item = str(item).strip().strip("'\"")
         
            if item.upper() in ['NONE', 'NULL', '']:
                continue
                
            if item.startswith("DESIGNATION:"):
                designation_buffer.append(item.replace("DESIGNATION:", "").strip())                
            elif len(item) == 2 and item.isalpha() and item.isupper():
                designation_buffer.append(item)
            else:
                if interleaving_type == "Applications":
                    if line_type == "dot":  # ✅ only add app numbers, not the parent itself
                        indent = "&nbsp;" * depth
                        bullet_items.append(f"{indent}• {item}")
                else:
                    if line_type == "comma":
                        bullet_items.append(f"<b>{item}</b>") # bullet_items.append(f"<b>{item[1:].strip()}</b>")
                    elif line_type == "dot":
                        indent = "&nbsp;" * depth  # optional: indent based on depth
                        bullet_items.append(f"{indent}• {item}") # bullet_items.append(f"• {item[1:].strip()}")
                    else:
                        bullet_items.append(item) # bullet_items.append(f"• {item}")
        
        if len(bullet_items) > 20:
            bullet_items = bullet_items[:20] + ["..."]
            
        # ✅ If we collected DESIGNATION states, add as one compact line
        if designation_buffer:
            bullet_items.append("• DESIGNATION: " + ", ".join(designation_buffer))
        
        # print("bullet_items:", bullet_items)
        
        tooltip = f"<b>{prefix}</b><br>"
        # print("1. tooltip:", tooltip)

        if interleaving_type == "Images":
            # Safely add images for every valid string in items
            for img in items:
                if isinstance(img, str) and img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    tooltip += f"<img src='{img}' width='400' style='max-height:600px; object-fit:contain;'><br>"
                else:
                    # If you want to add non-image strings as text, otherwise skip
                    tooltip += f"{img}<br>"
        elif bullet_items and line_type == "dot":
            tooltip += f"<i>{interleaving_type}:</i><br>" + "<br>".join(bullet_items)

        # print("2. tooltip:", tooltip)
        
        return tooltip

    def _prepare_sunburst_df(self, df, interleaving_type, reference_mode):
        """Shared preprocessing for single & multiple sunburst plots."""
        
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        # Define tooltip types
        tooltip_types = {
            "Applications", "Parents", "Priorities", "Publications", "Citations", 
            "Classifications", "Parties", "Legal Events", "Procedural Codes", 
            "Bibliographic Register Codes", "Events Register Codes", 
            "Procedural Steps", "UPP Codes"
        }
        # print("_prepare_sunburst_df 1.: interleaving_type, reference_mode):", interleaving_type, reference_mode)
        # Copy to avoid mutating original
        df = df.copy()
        df['interleaving_type'] = str(interleaving_type).capitalize()
        df['reference_mode'] = str(reference_mode).capitalize()
        df = df[df['id'].apply(lambda x: isinstance(x, str) and x.strip() != '')] 
        
        # ✅ Propagate event_date if present (for animations)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

        # Normalize type column and drop empty/non-string IDs
        df['type'] = df['type'].str.strip()
        
        # Prepare formatted IDs for labels
        df['formatted_id'] = df['id'].apply(lambda x: str(x).split(":")[0] if ": [" in str(x) else str(x))
        
        # create a cleaned key used for tooltip lookups from the original id (not formatted_id)
        df['id_clean'] = df['id'].apply(lambda x: ansi_escape.sub('', str(x)).strip())
        df['tooltip'] = df.get('tooltip', '')

        def normalize_key(s):
            return ansi_escape.sub('', str(s)).strip().split(' ', 1)[0]

        # --- Ensure full_tooltip_map exists ---
        if not hasattr(self, 'full_tooltip_map') or self.full_tooltip_map is None:
            self.full_tooltip_map = {}

        # print("_prepare_sunburst_df 2.: interleaving_type, reference_mode", interleaving_type, reference_mode)
        # Pre-populate full_tooltip_map from existing tooltips in this df
        for _, row in df.iterrows():
            key = normalize_key(row['formatted_id'])
            if row.get('tooltip') and key not in self.full_tooltip_map:
                self.full_tooltip_map[key] = {
                    'tooltip': row['tooltip'],
                    'interleaving_type': str(interleaving_type).capitalize(),
                    'reference_mode': str(reference_mode).capitalize()
                }
        
        # Assign tooltips for filtered DataFrame
        for idx, row in df.iterrows():
            key = normalize_key(row['id'])
            # Try restoring from persistent map
            tooltip_val = self.full_tooltip_map.get(key)
            if tooltip_val:
                df.at[idx, 'tooltip'] = tooltip_val
            else:
                df.at[idx, 'tooltip'] = self.build_tooltip(row) # , interleaving_type)
        
        # Propagate current application/publication IDs
        app_ids, pub_ids, current_app, current_pub = [], [], "", ""
        app_types = ("Applications", "Publications", "Parents", "Priorities") # can be extended if needed        
        for _, row in df.iterrows():
            typ = str(row.get('type', '')).strip()
            if typ in ("Publications", "Parents", "Priorities"):
                current_pub = row['formatted_id']
            # update current_app only if this row is an application/publication/parent/priority
            if typ in app_types:
                current_app = row['formatted_id']
            app_ids.append(current_app if current_app else row['formatted_id'])
            pub_ids.append(current_pub if current_pub else row['formatted_id'])
        df['app_id'] = app_ids
        df['pub_id'] = pub_ids
        
        # # --- DEBUG: Before restoration ---
        # print("💡 Tooltip column before restoration:")
        # print(df[['id', 'tooltip']].head(5))
    
        # 🔥 Restore tooltips from full tree if available
        restored_count = 0
        if getattr(self, 'full_tooltip_map') and self.full_tooltip_map:
            # Make sure keys match the filtered df
            for idx, row in df.iterrows():
                key = row['formatted_id']
                cached = self.full_tooltip_map.get(key)
                if cached:
                    df.at[idx, 'tooltip'] = cached['tooltip']
                    df.at[idx, 'interleaving_type'] = cached['interleaving_type']
                    df.at[idx, 'reference_mode'] = cached['reference_mode']
                    restored_count += 1                
                    
        # print(f"✅ Restored {restored_count} tooltips from full_tooltip_map (filtered DF has {len(df)} rows)")
    
        # # --- DEBUG: After restoration ---
        # print("💡 Tooltip column after restoration:")
        # print(df[['id', 'tooltip']].head(5))        
        # print("reference_mode, interleaving_type:", reference_mode, interleaving_type)
        
        # ✅ Generate tooltip only for rows missing one
        # Generate missing tooltips
        generated_count = 0
        for idx, row in df.iterrows():
            if pd.isna(row['tooltip']) or row['tooltip'] == '':
                if interleaving_type in tooltip_types:
                    df.at[idx, 'tooltip'] = self.build_tooltip(row) # , interleaving_type)
                elif interleaving_type == "Images":
                    df.at[idx, 'tooltip'] = f"<img src='{row['id']}' width='200'>" if str(row['id']).startswith("http") else row['id']
                else:
                    df.at[idx, 'tooltip'] = row['id']
                generated_count += 1

        # print(f"✅ Generated {generated_count} missing tooltips")
    
        # # --- DEBUG: Final tooltip snapshot ---
        # print("💡 Final tooltip snapshot (first 10 rows):")
        # print(df[['id', 'tooltip']].head(10))

        df = self.assign_branch_colors(df, fromWhere=1)
    
        # --- KEY PART: make node ids unique while preserving parent links ---
        # Reset index to get stable numeric row IDs
        df = df.reset_index(drop=True)
        
        # Unique ids and parent mapping
        df['uid'] = df.index.astype(str) + '|' + df['id'].astype(str)

        # Build a mapping from original id -> list of indices where it appears
        id_to_indices = {}
        for idx, orig_id in df['id'].items():
            id_to_indices.setdefault(orig_id, []).append(idx)
            
        # Compute parent_uid for every row.
        # We'll map parent string -> the first row index that has that original id.
        # This works when parent row is present in the DataFrame (typical for a tree built from the file).
        def find_parent_uid(parent_val):
            matches = id_to_indices.get(parent_val, [])
            return df.at[matches[0], 'uid'] if matches else ""
        
        df['parent_uid'] = df['parent'].apply(find_parent_uid)

        # Ensure numeric values
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(1)

        return df
    
    def plot_sunburst(self, df, interleaving_type, reference_mode, output_path=None):
        if df is None or not hasattr(df, "empty") or df.empty:
            print("No data to plot.")
            return None, None
            
        # reference_mode = df['reference_mode'].iloc[0] if 'reference_mode' in df.columns else 'Application'
        # interleaving_type = df['interleaving_type'].iloc[0] if 'interleaving_type' in df.columns else 'no further selection'
        # print("plot_sunburst: reference_mode, interleaving_type, output_path:", reference_mode, interleaving_type, output_path)
        
        df = self._prepare_sunburst_df(df, interleaving_type, reference_mode)  # ✅ factor out shared logic
        
        # Right before setting hovertemplate:

            
        # Build both hovertemplate variants
        hovertemplate_non_images = f"<b>{reference_mode}:</b> %{{customdata[1]}}<extra></extra>"
        hovertemplate_images     = "%{customdata[0]}<extra></extra>" 
        # Choose the one to use
        hovertemplate = hovertemplate_images if interleaving_type == "Images" else hovertemplate_non_images
        # print("hovertemplate:", hovertemplate)

        # print("hovertemplate:", hovertemplate)
        fig = go.Figure(
            go.Sunburst(
                ids=df['uid'],
                labels=df['formatted_id'], 
                text=df['formatted_id'],
                parents=df['parent_uid'],
                values=df['value'],
                customdata=np.stack([df['app_id'], df['tooltip'], df['hover_bg']], axis=-1),
                hovertemplate=hovertemplate,                
                branchvalues='remainder' if interleaving_type == "Priorities" else 'total',
                marker=dict(colors=df['color'], line=dict(color='white', width=0.5)) 
            )
        )
    
        fig.update_traces(
            textinfo='text',
            textfont_size=12,
            hoverlabel=dict(
                bgcolor=None,
                font_color="black",
                bordercolor="rgba(0,0,0,0.1)"
            ),
        )
        
        fig.update_layout(
            title=dict(
                text=f'Patent Tree: {interleaving_type}',
                x=0.5,
                xanchor='center',
            ),            
            margin=dict(t=30, l=0, r=0, b=0),
            width=1200,
            height=900,
            showlegend=False
        )        
                
        if output_path is None:
            output_path = os.path.join(self.workdir, "filtered_tree.html")
        fig.write_html(output_path, include_plotlyjs='cdn') # Save silently (no auto display)
        return fig, output_path
        
    def plot_sunburst_over_time(self, df, interleaving_type, reference_mode, output_path=None, time_col="event_date"):
        """
        Create an animated sunburst that evolves over time.
        Each frame shows the tree state up to the current timestamp.
        """
        # --- Prepare the input dataframe ---
        df = self._prepare_sunburst_df(df, interleaving_type, reference_mode)
        if time_col not in df.columns:
            # print(f"⚠️ Column '{time_col}' not found — cannot animate.")
            return self.plot_sunburst(df, interleaving_type, reference_mode, output_path)

        # print("🧪 event_date sample:", df[time_col].head(10).tolist())
        # print("🧩 label sample:", df.get('formatted_id', df.get('id')).head(10).tolist())

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        # Structural rows (Applications, Parents, etc.) may have NaT for event_date
        # because they are not events themselves. Fill them with the earliest known
        # event date so they always appear as parent anchors in every animation frame
        # instead of being dropped — which would orphan all their legal event children.
        earliest = df[time_col].dropna().min()
        if pd.isna(earliest):
            print(f"\u26a0\ufe0f No valid timestamps in '{time_col}'. Showing static chart.")
            return self.plot_sunburst(df, interleaving_type, reference_mode, output_path)
        df[time_col] = df[time_col].fillna(earliest)

        # One frame per calendar day — same-day events produce identical frames
        df["_date_only"] = df[time_col].dt.normalize()
        timestamps = sorted(df["_date_only"].unique())

        if not timestamps:
            print(f"\u26a0\ufe0f No valid timestamps in '{time_col}'. Showing static chart.")
            return self.plot_sunburst(df, interleaving_type, reference_mode, output_path)

        # print("Unique timestamps:", timestamps)
        # print("Total rows for first timestamp:", len(df[df[time_col] <= timestamps[0]]))
        # print("Sample event_date values:\n", df[time_col].head(10))
        
        # Build both hovertemplate variants
        hovertemplate_non_images = f"<b>{reference_mode}:</b> %{{customdata[1]}}<extra></extra>"
        hovertemplate_images     = "%{customdata[0]}<extra></extra>" 
        # Choose the one to use
        hovertemplate = hovertemplate_images if interleaving_type == "Images" else hovertemplate_non_images
        # print("hovertemplate:", hovertemplate)
        
        # --- Build animation frames ---
        # customdata + hovertemplate must be in every frame: Plotly sunburst does a
        # full data replacement per frame and does NOT retain the initial trace hover data.
        frames = []
        branchvals = 'remainder' if interleaving_type == "Priorities" else 'total'
        for t in timestamps:
            subset = df[df["_date_only"] <= t]
            frames.append(
                go.Frame(
                    data=[go.Sunburst(
                        ids=subset["uid"],
                        labels=subset["formatted_id"],
                        parents=subset["parent_uid"],
                        values=subset["value"],
                        branchvalues=branchvals,
                        marker=dict(colors=subset["color"], line=dict(color='white', width=0.5)),
                        customdata=np.stack([subset["app_id"], subset["tooltip"], subset["hover_bg"]], axis=-1),
                        hovertemplate=hovertemplate,
                    )],
                    name=str(t.date() if hasattr(t, "date") else t)
                )
            )

        # --- Initial figure ---
        first_subset = df[df["_date_only"] <= timestamps[0]]
        fig = go.Figure(
            data=[go.Sunburst(
                ids=first_subset["uid"],
                labels=first_subset["formatted_id"],
                text=first_subset['formatted_id'], 
                # text=first_subset['tooltip'], 
                parents=first_subset["parent_uid"],
                values=first_subset["value"],
                # hovertext=first_subset["tooltip"],
                customdata=np.stack([first_subset["app_id"], first_subset["tooltip"], first_subset["hover_bg"]], axis=-1),
                hovertemplate=hovertemplate, 
                branchvalues='remainder' if interleaving_type == "Priorities" else 'total', 
                marker=dict(colors=first_subset["color"], line=dict(color='white', width=0.5)),
                domain=dict(x=[0.05, 0.95], y=[0.18, 1.0]),                
            )],
            frames=frames
        )

        # --- Layout and controls ---
        fig.update_layout(
            #title=f"Patent Tree over Time: {interleaving_type}",
            title=dict(
                text=f"Patent Tree over Time: {interleaving_type}",
                x=0.5,           # horizontally centered
                xanchor='center',
            ),            
            margin=dict(t=60, l=20, r=20, b=100),  # more bottom margin for the slider
            autosize=True,       # fill whatever container width is given
            width=None,          # don't hardcode — patent_processor sets this
            # width=1200,
            height=900,
            uirevision="persist",
            transition={"duration": 400, "easing": "cubic-in-out"},
            updatemenus=[{
                "type": "buttons",
                "x": -0.02, "xanchor": "left",   # centered horizontally
                "y": -0.08, "yanchor": "top",      # sit just below the chart domain                
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 1000, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 500}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "x": 0.18,  "xanchor": "left",    # aligned to chart left edge
                "y": -0.06, "yanchor": "top",     # below the buttons
                "len": 0.80,                      # nearly full width                
                "currentvalue": {
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "center",
                    "font": {"size": 12},
                },                
                "steps": [
                    {"args": [[str(t.date() if hasattr(t, "date") else t)],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate"}],
                     "label": str(t.date() if hasattr(t, "date") else t),
                     "method": "animate"}
                    for t in timestamps
                ]
            }]
        )

        # --- Output ---
        if output_path is None:
            output_path = os.path.join(self.workdir, "animated_tree.html")

        fig.write_html(output_path, include_plotlyjs='cdn')
        # print(f"✅ Animated sunburst saved to: {output_path}")
        return fig, output_path

    def plot_sunburst_for_widget(self, df, interleaving_type, reference_mode, output_path=None, time_col="event_date"):
        """
        Like plot_sunburst_over_time, but returns the figure WITHOUT built-in Plotly
        animation controls (no updatemenus, no sliders). The caller (patent_processor)
        adds real ipywidgets Play/Pause/Slider controls that live inside the HBox frame.

        Returns
        -------
        fig          : go.Figure  (with frames, no Plotly controls)
        timestamps   : list of pd.Timestamp
        df_prepared  : pd.DataFrame with uid/parent_uid/color/… columns ready for reuse
        output_path  : str
        hovertemplate: str
        """
        df = self._prepare_sunburst_df(df, interleaving_type, reference_mode)
        if time_col not in df.columns:
            fig, path = self.plot_sunburst(df, interleaving_type, reference_mode, output_path)
            return fig, [], df, path, ""

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        # Structural rows (Applications, Parents, etc.) have NaT event_date.
        # Fill with earliest known event date so they always appear as parent anchors
        # in every animation frame, preventing their legal-event children from being orphaned.
        earliest = df[time_col].dropna().min()
        if pd.isna(earliest):
            fig, path = self.plot_sunburst(df, interleaving_type, reference_mode, output_path)
            return fig, [], df, path, ""
        df[time_col] = df[time_col].fillna(earliest)

        # Deduplicate to one frame per calendar day — multiple events on the
        # same day produce identical-looking frames and inflate serialisation cost.
        df["_date_only"] = df[time_col].dt.normalize()
        timestamps = sorted(df["_date_only"].unique())

        if not timestamps:
            fig, path = self.plot_sunburst(df, interleaving_type, reference_mode, output_path)
            return fig, [], df, path, ""

        hovertemplate = (
            "%{customdata[0]}<extra></extra>"
            if interleaving_type == "Images"
            else f"<b>{reference_mode}:</b> %{{customdata[1]}}<extra></extra>"
        )

        # --- Build frames ---
        # customdata + hovertemplate must be in every frame: Plotly sunburst does a
        # full data replacement per frame and does NOT retain the initial trace hover data.
        frames = []
        branchvals = 'remainder' if interleaving_type == "Priorities" else 'total'
        for t in timestamps:
            subset = df[df["_date_only"] <= t]
            frames.append(go.Frame(
                data=[go.Sunburst(
                    ids=subset["uid"],
                    labels=subset["formatted_id"],
                    parents=subset["parent_uid"],
                    values=subset["value"],
                    branchvalues=branchvals,
                    marker=dict(colors=subset["color"], line=dict(color='white', width=0.5)),
                    customdata=np.stack([subset["app_id"], subset["tooltip"], subset["hover_bg"]], axis=-1),
                    hovertemplate=hovertemplate,
                )],
                name=str(t.date() if hasattr(t, "date") else t)
            ))

        # --- Initial figure (NO updatemenus / sliders — those go in the widget layer) ---
        first_subset = df[df["_date_only"] <= timestamps[0]]
        fig = go.Figure(
            data=[go.Sunburst(
                ids=first_subset["uid"],
                labels=first_subset["formatted_id"],
                # text=first_subset["tooltip"],
                text=first_subset["formatted_id"],
                parents=first_subset["parent_uid"],
                values=first_subset["value"],
                # hovertext=first_subset["tooltip"],
                customdata=np.stack([first_subset["app_id"], first_subset["tooltip"], first_subset["hover_bg"]], axis=-1),
                hovertemplate=hovertemplate,
                branchvalues='remainder' if interleaving_type == "Priorities" else 'total',
                marker=dict(colors=first_subset["color"], line=dict(color='white', width=0.5)),
            )],
            frames=frames
        )

        fig.update_layout(
            title=dict(
                text=f"Patent Tree over Time: {interleaving_type}",
                x=0.5, xanchor="center",
            ),
            margin=dict(t=50, l=10, r=10, b=80),  # b=80 just enough for one control row
            autosize=True,
            height=820,
            uirevision="persist",
            transition={"duration": 400, "easing": "cubic-in-out"},
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "x": 0.02, "xanchor": "left",
                "y": -0.06, "yanchor": "top",
                "direction": "right",
                "pad": {"r": 8, "t": 0},
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 1000, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 500}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "x": 0.18, "xanchor": "left",   # immediately right of buttons
                "y": -0.04, "yanchor": "top",    # same row as buttons
                "len": 0.80,
                "pad": {"t": 0, "b": 0},
                "currentvalue": {
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "center",
                    "font": {"size": 12, "color": "#333"},
                    "offset": 20,
                },
                "ticklen": 4,
                "minorticklen": 2,
                "steps": [
                    {"args": [[str(t.date() if hasattr(t, "date") else t)],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate"}],
                     "label": str(t.date() if hasattr(t, "date") else t),
                     "method": "animate"}
                    for t in timestamps
                ],
            }],
        )

        # Save HTML with fixed size (looks good when opened externally)
        if output_path is None:
            output_path = os.path.join(self.workdir, "animated_tree.html")
        fig.update_layout(autosize=False, width=1200, height=900)
        fig.write_html(output_path, include_plotlyjs='cdn')
        fig.update_layout(autosize=True, width=None, height=820)  # restore for widget

        return fig, timestamps, df, output_path, hovertemplate

    def plot_multiple_sunbursts(self, dfs, interleaving_types, reference_mode, output_path=None):
        """
        Plot multiple Sunburst trees in a single figure.
    
        Parameters
        ----------
        tree_dfs : list of pd.DataFrame
            List of DataFrames, one per tree.
        interleaving_types : list of str
            List of interleaving_type strings, one per tree.
        output_path : str, optional
            Path to save HTML output. If None, uses default.
        """

        n = len(dfs)
        if n == 0:
            print("No dataframes provided.")
            return None, None
            
        cols = min(n, 3)  # max 3 trees per row
        rows = (n + cols - 1) // cols
        
        # vertical_spacing is a fraction of the total figure height allocated between rows.
        # Domain-type subplots (sunbursts) don't auto-shrink, so we tighten it manually.
        # For a single row there is no gap to worry about.
        v_spacing = 0.02 if rows > 1 else 0.0

        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=interleaving_types,
            vertical_spacing=v_spacing,
        )
        
        for i, (df, interleaving_type) in enumerate(zip(dfs, interleaving_types)):
            if df is None or not hasattr(df, "empty") or df.empty:
                continue

            df = self._prepare_sunburst_df(df, interleaving_type, reference_mode)

            # Hovertemplate switch
            hovertemplate_non_images = f"<b>{reference_mode}:</b> %{{customdata[1]}}<extra></extra>"
            hovertemplate_images     = "%{customdata[0]}<extra></extra>"
            hovertemplate = hovertemplate_images if interleaving_type == "Images" else hovertemplate_non_images
            
            row = i // cols + 1
            col = i % cols + 1

            # # Ensure root exists
            # if not (df['parent'] == "").any():
            #     # Pick the first id as root if none exists
            #     root_id = df.iloc[0]['id']
            #     df.loc[df['id'] == root_id, 'parent'] = ""

            # Ensure numeric values
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(1)

            # print(df[['id', 'parent', 'value', 'type']].head(10))  # debug

            fig.add_trace(
                go.Sunburst(
                    ids=df['uid'],
                    labels=df['formatted_id'],
                    text=df['formatted_id'],
                    parents=df['parent_uid'],
                    values=df['value'],
                    hoverinfo='none',
                    branchvalues='remainder' if interleaving_type == "Priorities" else 'total',
                    marker=dict(colors=df['color'], line=dict(color='white', width=0.5))
                ), 
                row=row, 
                col=col
            )
    
        fig.update_traces(
            textinfo='text',
            textfont_size=12,
            hoverinfo='none',
        )
    
        # 420 px per row is enough for sunburst charts; the tighter vertical_spacing
        # above removes the bulk of the blank band between rows.
        fig.update_layout(
            width=500 * cols,
            height=420 * rows,
            showlegend=False,
            title_text="Patent Trees",
            margin=dict(t=60, b=20, l=20, r=20),
        )
    
        if output_path is None:
            output_path = "filtered_multi_tree.html"
    
        fig.write_html(output_path, include_plotlyjs='cdn')
        return fig, output_path
            
    def append_image_hover_script(self, html_path):
        custom_hover_script = """
        <style>
        #hover-img-preview {
            position: absolute;
            display: none;
            border: 1px solid #aaa;
            background: white;
            padding: 2px;
            z-index: 9999;
        }
        </style>
        <div id="hover-img-preview">
            <img id="hover-img" src="" width="600">
        </div>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const hoverBox = document.getElementById('hover-img-preview');
            const hoverImg = document.getElementById('hover-img');
            const myPlot = document.querySelector('.js-plotly-plot');
            if (myPlot && Plotly && Plotly.Plots) {
                myPlot.on('plotly_hover', function(data){
                    var tooltip = data.points[0].customdata?.[0] || '';
                    var match = tooltip.match(/src=['"]([^'"]+)['"]/);
                    if (match) {
                        hoverImg.src = match[1];
                        hoverBox.style.left = (data.event.clientX + 10) + 'px';
                        hoverBox.style.top = (data.event.clientY + 10) + 'px';
                        hoverBox.style.display = 'block';
                    }
                });
                myPlot.on('plotly_unhover', function(){
                    hoverBox.style.display = 'none';
                    hoverImg.src = '';
                });
            }
        });
        </script>
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', custom_hover_script + '</body>')
        else:
            html_content += custom_hover_script
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def run_divitree_plotter(self, tree_type="ORAP", process_both=False,
                             reference_mode="Application", interleaving_type=None, tree_path=None):
        """
        Plot the latest tree output that matches the requested tree_type.
        Args:
            tree_type (str|None): "ORAP" or "priority".
            tree_path (str|None): If given, plot this specific file directly.
        """
        if tree_path and os.path.exists(tree_path):
            all_paths = [tree_path]
        else:
            all_paths = sorted(
                [p for p in self.find_latest_output_files() if p],
                key=lambda p: os.path.getmtime(p),
                reverse=True
            )

        if not all_paths:
            print("❌ No tree output files found.")
            return None, None

        for tree_path in all_paths:
            if not tree_path or not isinstance(tree_path, str):
                continue
                
            filename_lower = os.path.basename(tree_path).lower()
            # print("filename_lower:", filename_lower)
                
            # print("tree_path:", tree_path)            
            tree_data, interleaving_type, reference_mode = self.read_tree_data(tree_path, interleaving_type, reference_mode)
            # print("run_divitree_plotter: interleaving_type, reference_mode:", interleaving_type, reference_mode)
            
            df = self.create_grouped_items_df(tree_data)
            if df.empty:
                print(f"⚠️ DataFrame is empty after parsing {tree_path}, skipping...")
                continue
                
            df['interleaving_type'] = str(interleaving_type).capitalize()
            df['reference_mode'] = str(reference_mode).capitalize()
            
            html_filename = f"sunburst_plot_{interleaving_type}.html"
            html_path = os.path.join(self.workdir, html_filename)

            try:
                fig, _ = self.plot_sunburst(df, interleaving_type, reference_mode, html_path)
            except Exception as e:
                print(f"❌ Error generating Plotly chart: {e}")
                continue

            html_header = f"""
                ✅ Chart saved to: <code>{html_path}</code><br>
                📂 To open it in a new browser tab, right-click the file in the left sidebar and choose:<br>
                <b><i>+ Open in New Browser Tab</i></b>
                """
                
            if interleaving_type == "Images":
                self.append_image_hover_script(html_path)

            return fig, html_path, html_header  # Only process one file and return immediately

        print("❌ No tree files processed.")
        return None, None, None