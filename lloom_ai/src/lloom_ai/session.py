# Concept induction session functions
# =================================================

# Imports
import time
import pandas as pd
import ipywidgets as widgets
import random

# TODO(MSL): check on relative imports in package
# Local imports
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from concept_induction import *
    from concept import Concept
else:
    # uses current package visibility
    from .concept_induction import *
    from .concept import Concept

# SESSION class ================================
class Session:
    def __init__(
        self,
        in_df: pd.DataFrame,
        doc_id_col: str,
        doc_col: str,
        save_path: str = None,
        debug: bool = False,
    ):
        # Settings
        self.model_name = "gpt-3.5-turbo"
        self.synth_model_name = "gpt-4"
        self.use_base_api = True
        self.debug = debug  # Whether to run in debug mode

        if save_path is None:
            # Automatically set using timestamp
            t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            save_path = f"./exports/{t}"
        self.save_path = save_path

        # Input data
        self.in_df = in_df
        self.doc_id_col = doc_id_col
        self.doc_col = doc_col
        self.df_to_score = in_df  # Default to in_df for concept scoring

        # Output data
        self.saved_dfs = {}  # maps from (step_name, time_str) to df
        self.concepts = {}  # maps from concept_id to Concept 
        self.results = {}  # maps from concept_id to its score_df
        self.df_filtered = None  # Current quotes df
        self.df_bullets = None  # Current bullet points df
        
        # Cost/Time tracking
        self.time = {}  # Stores time required for each step
        self.cost = []  # Stores result of cost estimation
        self.tokens = {
            "in_tokens": [],
            "out_tokens": [],
        }
    
    def save(self):
        # Saves current session to file
        t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        cur_path = f"{self.save_path}__{t}.pkl"
        with open(cur_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved session to {cur_path}")

    def get_save_key(self, step_name):
        t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        k = (step_name, t)  # Key of step name and current time
        return k
    
    def show_session_summary(self):
        # Time
        total_time = np.sum(list(self.time.values()))
        print(f"Total time: {total_time:0.2f} sec ({(total_time/60):0.2f} min)")
        for step_name, time in self.time.items():
            print(f"\t{step_name}: {time:0.2f} sec")

        # Cost
        total_cost = np.sum(self.cost)
        print(f"\n\nTotal cost: {total_cost:0.2f}")

        # Tokens
        in_tokens = np.sum(self.tokens["in_tokens"])
        out_tokens = np.sum(self.tokens["out_tokens"])
        total_tokens =  in_tokens + out_tokens
        print(f"\n\nTokens: total={total_tokens}, in={in_tokens}, out={out_tokens}")

    def print_active_concepts(self):
        active_concepts = self.__get_active_concepts()
        print(f"Active concepts (n={len(active_concepts)}):")
        for c_id, c in active_concepts.items():
            print(f"- {c.name}: {c.prompt}")

    # HELPER FUNCTIONS ================================
    async def gen(self, seed=None, args=None, debug=True):
        # TODO: modify to automatically determine args
        if args is None:
            args = {
                "filter_n_quotes": 2,
                "summ_n_bullets": "2-4",
                "cluster_batch_size": 20,
                "synth_n_concepts": 10,
            }

        # Run concept generation
        df_filtered = await distill_filter(
            text_df=self.in_df, 
            doc_col=self.doc_col,
            doc_id_col=self.doc_id_col,
            model_name=self.model_name,
            n_quotes=args["filter_n_quotes"],
            seed=seed,
            sess=self,
        )
        self.df_to_score = df_filtered
        self.df_filtered = df_filtered
        if debug:
            print("df_filtered")
            display(df_filtered)
        
        df_bullets = await distill_summarize(
            text_df=df_filtered, 
            doc_col=self.doc_col,
            doc_id_col=self.doc_id_col,
            model_name=self.model_name,
            n_bullets=args["summ_n_bullets"],
            seed=seed,
            sess=self,
        )
        self.df_bullets = df_bullets
        if debug:
            print("df_bullets")
            display(df_bullets)

        df_cluster = await cluster(
            text_df=df_bullets, 
            doc_col=self.doc_col,
            doc_id_col=self.doc_id_col,
            batch_size=args["cluster_batch_size"],
            sess=self,
        )
        if debug:
            print("df_cluster")
            display(df_cluster)
        
        df_concepts = await synthesize(
            cluster_df=df_cluster, 
            doc_col=self.doc_col,
            doc_id_col=self.doc_id_col,
            model_name=self.synth_model_name,
            n_concepts=args["synth_n_concepts"],
            pattern_phrase="unique topic",
            seed=seed,
            sess=self,
        )
        if debug:
            # Print results
            print("synthesize")
            for k, c in self.concepts.items():
                print(f'- Concept {k}:\n\t{c.name}\n\t- Prompt: {c.prompt}')

    def __get_selection_widget(self, options_dict):
        # Widget with a search field and lots of checkboxes
        search_widget = widgets.Text()
        output_widget = widgets.Output()
        options = [x for x in options_dict.values()]
        options_layout = widgets.Layout(
            overflow='auto',
            border='1px solid grey',
            height='300px',
            flex_flow='column',
            display='flex',
        )
        
        options_widget = widgets.VBox(options, layout=options_layout)
        multi_select = widgets.VBox([search_widget, options_widget])

        @output_widget.capture()
        def on_checkbox_change(change):
            c_descriptions = [c.description for c in options_widget.children if c.value]
            selected_concepts = {c_id: c for c_id, c in self.concepts.items() if self.format_desc(c) in c_descriptions}
            for c_id, c in self.concepts.items():
                c.active = (c_id in selected_concepts)
            
        for checkbox in options:
            checkbox.observe(on_checkbox_change, names="value")

        # Wire the search field to the checkboxes
        @output_widget.capture()
        def on_text_change(change):
            search_input = change['new']
            if search_input == '':
                # Reset search field
                new_options = sorted(options, key = lambda x: x.value, reverse = True)
            else:
                # Filter by search field using difflib.
                c_descriptions = [c.description for c in options_widget.children]
                close_matches = [x for x in c_descriptions if str.lower(search_input.strip('')) in str.lower(x)]
                new_options = sorted(
                    [x for x in options if x.description in close_matches], 
                    key = lambda x: x.value, reverse = True
                )
            options_widget.children = new_options

        search_widget.observe(on_text_change, names='value')
        display(output_widget)
        return multi_select

    def format_desc(self, c):
        return f"{c.name}:\n{c.prompt}"
    
    def select(self):
        options_dict = {
            c.name: widgets.Checkbox(
                description=self.format_desc(c),
                value=c.active,
                style={"description_width":"0px"},
                layout=widgets.Layout(
                    width='100%',
                    text_overflow='initial',
                    overflow='visible',
                ),
            ) for c_id, c in self.concepts.items()
        }
        ui = self.__get_selection_widget(options_dict)
        return ui

    def __get_active_concepts(self):
        return {c_id: c for c_id, c in self.concepts.items() if c.active}

    # Score the specified concepts
    # Only score the concepts that are active
    async def score(self, c_ids=None, get_highlights=True):
        concepts = {}
        active_concepts = self.__get_active_concepts()
        if c_ids is None:
            # Score all active concepts
            for c_id, c in active_concepts.items():
                concepts[c_id] = c
        else:
            # Score only the specified concepts
            for c_id in c_ids:
                if c_id in active_concepts:
                    concepts[c_id] = active_concepts[c_id]
        
        # Run usual scoring
        score_df = await score_concepts(
            text_df=self.df_to_score, 
            text_col=self.doc_col, 
            doc_id_col=self.doc_id_col,
            concepts=concepts,
            get_highlights=get_highlights,
            sess=self,
        )
        # Store results for each concept
        for c_id in concepts.keys():
            self.results[c_id] = score_df[score_df['concept_id'] == c_id]
        return score_df

    def __get_concept_from_name(self, name):
        if name == "Outlier":
            return Concept(name="Outlier", prompt=OUTLIER_CRITERIA, example_ids=[], active=True)
        for c_id, c in self.concepts.items():
            if c.name == name:
                return c
        return None
    
    def get_score_df(self):
        active_concepts = self.__get_active_concepts()
        active_score_dfs = [self.results[c_id] for c_id in active_concepts.keys() if c_id in self.results]
        score_df = pd.concat(active_score_dfs)
        score_df = score_df.rename(columns={"doc_id": self.doc_id_col})
        return score_df

    def __get_concept_highlights(self, c, threshold=0.75, highlight_col="highlight", lim=3):
        if c.name == "Outlier":
            return []
        if c.id not in self.results:
            return []
        score_df = self.results[c.id].copy()
        score_df = score_df[score_df["score"] > threshold]
        highlights = score_df[highlight_col].tolist()
        # shuffle highlights
        random.shuffle(highlights)
        if lim is not None:
            highlights = highlights[:lim]
        return highlights

    def __get_rep_examples(self, c):
        if c.name == "Outlier":
            return []
        if c.id not in self.results:
            return []
        score_df = self.results[c.id].copy()
        ex_ids = c.example_ids
        ex = score_df[score_df["doc_id"].isin(ex_ids)]["highlight"].tolist()
        return ex

    def __get_prevalence_df(self, item_df, threshold=0.75, include_outliers=False):
        matched = item_df[(item_df.concept_score_orig > threshold)]
        if not include_outliers:
            matched = matched[item_df.concept != "Outlier"]

        df = matched.groupby(by=["id", "concept"]).count().reset_index()[["concept", self.doc_col]]
        concepts = [self.__get_concept_from_name(c_name) for c_name in df.concept.tolist()]
        df["criteria"] = [c.prompt for c in concepts]
        df["rep_examples"] = [self.__get_rep_examples(c) for c in concepts]
        df["highlights"] = [self.__get_concept_highlights(c, threshold) for c in concepts]
        df = df.rename(columns={self.doc_col: "n_matches"})
        df["prevalence"] = np.round(df["n_matches"] / len(self.in_df), 2)
        df = df[["concept", "criteria", "prevalence", "highlights", "rep_examples"]]
        return df
        

    def vis(self, cols_to_show=[], custom_groups={}, show_highlights=True, norm_by="concept", export_df=False):
        active_concepts = self.__get_active_concepts()
        score_df = self.get_score_df()

        widget, matrix_df, item_df, item_df_wide = visualize(
            in_df=self.in_df,
            score_df=score_df,
            doc_col=self.doc_col,
            doc_id_col=self.doc_id_col,
            score_col="score",
            df_filtered=self.df_filtered,
            df_bullets=self.df_bullets,
            concepts=active_concepts,
            cols_to_show=cols_to_show,
            custom_groups=custom_groups,
            show_highlights=show_highlights,
            norm_by=norm_by,
        )
        if export_df:
            return self.__get_prevalence_df(item_df)
        
        return widget

    async def summarize(self, model_name="gpt-4"):
        score_df = self.get_score_df()
        summary_df = await summarize_concept(score_df, self.in_df, model_name, self.doc_id_col)
        prevalence_df = self.vis(export_df=True)[["concept", "rep_examples", "highlights"]]
        out_df = summary_df.merge(prevalence_df, left_on="concept_name", right_on="concept", how="left")
        return out_df[["concept_name", "summary", "rep_examples", "concept_prompt", "n_matches", "highlights"]]

    async def add(self, name, prompt, ex_ids=[], get_highlights=True):
        # Add concept
        c = Concept(name=name, prompt=prompt, example_ids=ex_ids, active=True)
        self.concepts[c.id] = c

        # Run scoring
        cur_score_df = await self.score(c_ids=[c.id], get_highlights=get_highlights)
        
        # Store results
        self.results[c.id] = cur_score_df
    
    
    async def edit(self):
        raise NotImplementedError("Edit function not yet implemented")