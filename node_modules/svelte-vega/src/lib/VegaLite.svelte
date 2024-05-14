<script lang="ts">
	import type { EmbedOptions, Mode } from 'vega-embed';
	import type { SignalListeners, VisualizationSpec, View } from './types';
	import VegaEmbed from './VegaEmbed.svelte';

	export let spec: VisualizationSpec;
	export let options: EmbedOptions = {};
	export let data: Record<string, unknown> = {};
	export let signalListeners: SignalListeners = {};
	export let view: View | undefined = undefined;

	const mode = 'vega-lite' as Mode;
	$: vegaLiteOptions = { ...options, mode: mode };
</script>

<VegaEmbed
	{spec}
	{data}
	{signalListeners}
	options={vegaLiteOptions}
	bind:view
	on:onNewView
	on:onError
/>
