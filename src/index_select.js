import ConceptSelectView from "./components/ConceptSelectView.svelte";

export function render({ model, el }) {
	let v = new ConceptSelectView({ target: el, props: { model} });
	return () => v.$destroy();
}
