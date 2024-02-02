import "./style.css";

import MatrixView from "./components/MatrixView.svelte";

export function render({ model, el }) {
	let mat = new MatrixView({ target: el, props: { model, el } });
	return () => mat.$destroy();
}
