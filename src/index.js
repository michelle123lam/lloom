import MatrixView from "./components/MatrixView.svelte";

function render({ model, el }) {
	let mat = new MatrixView({ target: el, props: { model, el } });
	return () => mat.$destroy();
}

export default {render}
