<script>
    // Component to render matrix in notebook
    // Imports
    import { onMount } from "svelte";
    import { createEventDispatcher } from 'svelte';
    import * as d3 from "https://esm.sh/d3@7";  // notebook version
    // import * as d3 from "d3";  // web app version
	

    // Properties
    export let data;  // data for matrix
    export let div;  // div containing the matrix
    export let numConcepts;  // total number of concepts (for dynamic sizing)
    export let numSlices; // total number of slices (for dynamic sizing)

    let chart;
    const dispatch = createEventDispatcher();
    let rowLabelWidth = 200;
    let rowLabelHeight = 150;
    let perSliceWidth = 60;
    let colLabelHeight = 300;
    let perConceptHeight = 60;
    
    onMount(() => {
        renderMatrix(data);
	});

    // Create the matrix visualization using the data
	function renderMatrix(data_in) {
        if (data_in != undefined && numConcepts > 0) {
            let data_json = JSON.parse(data_in);
            const width = rowLabelWidth + (perSliceWidth * numSlices);
            const height = colLabelHeight + (perConceptHeight * numConcepts);
            const cur_color = "#82C1FB";
            chart = new BubbleMatrix(div)
                .size([width, height])
                .options({
                    highlightScope: "matrix",
                    showSlider: false,
                    preserveRowOrder: true,
                    preserveColumnOrder: true,
                    sortOnAxisClick: false,
                })
                .font({
                    family: "system-ui",
                    size: 10,
                })
                .columns({
                    row: "concept",
                    column: "id",
                    value: "value",
                    detail: "example",
                })
                .colors({
                    above: cur_color,
                    below: cur_color,
                    row: "#eee",
                })
                .data(data_json);
            chart.render();	
        }
	}

    class BubbleMatrix {
        constructor(container = document.createElement("DIV")) {
            this._container = container;

            this._options = new BubbleMatrixOptions()
            this.partitions = new Partitions(this);

            this._dataset = null;
            this._fieldNames = new FieldNames();
            this.chartData = null;

            this.measures = new Measures(this._getMeasureSvg());
            this.scales = new Scales();

            this.renderer = new Renderer(this);
        }

        get fieldNames() { return this._fieldNames; }

        size(_) {
            if (arguments.length) {
                this.measures.width = _[0];
                this.measures.height = _[1];
                return this;
            }
            else {
                return [this.measures.width, this.measures.height];
            }
        }

        colors(_) {
            if (arguments.length) {
                this.renderer.colors.row = _.row;
                this.renderer.colors.above = _.above;
                this.renderer.colors.below = _.below;
                this.renderer.colors.label = _.label;
                return this;
            }
            else {
                return this.renderer.colors;
            }
        }

        margin(_) {
            if (arguments.length) {
                this.measures.margin.left = _.left;
                this.measures.margin.right = _.right;
                this.measures.margin.top = _.top;
                this.measures.margin.bottom = _.bottom;
                return this;
            }
            else {
                return this.measures.margin;
            }
        }

        options(_) {
            return arguments.length ? (this._options = Object.assign(this._options, _), this) : this._options;
        }

        columns(_) {
            if (arguments.length) {
                this._fieldNames.column = _.column;
                this._fieldNames.row = _.row;
                this._fieldNames.value = _.value;
                this._fieldNames.detail = _.detail;
                return this;
            }
            else {
                return this._fieldNames;
            }
        }

        font(_) {
            if (arguments.length) {
                this.measures.font.family(_.family);
                this.measures.font.size(_.size);
                return this;
            }
            else {
                return this.measures.font;
            }
        }

        data(_) {
            return arguments.length ? (this._dataset = _, this) : this._dataset;
        }

        events(_) {
            if (arguments.length) {
                this.renderer.rows.onclick = _.onclick;
                this.renderer.rows.onhover = _.onhover;
                this.renderer.rows.oncancel = _.oncancel;
                return this;
            }
            else {
                return {
                    onclick: this.renderer.onclick,
                    onhover: this.renderer.onhover,
                    oncancel: this.renderer.oncancel
                }
            }
        }

        render() {
            const options = this._options;

            // const detached = !this._container.isConnected;
            // if (detached) document.body.append(this._container);

            this.chartData = new ChartData(this._dataset, this._fieldNames);
            this.chartData.numberIsPercentage = options.numberIsPercentage;
            this.chartData.preserveColumnOrder = options.preserveColumnOrder;
            this.chartData.preserveRowOrder = options.preserveRowOrder;
            this.chartData.numOfTopBottom = options.numberOfTopBottom;
            this.chartData.process();

            this.measures.initialize(this.chartData, options.showSlider);
            this.scales.initialize(this);

            this.partitions.initialize();
            this._container.appendChild(this.partitions.chartArea);

            this.renderer.rows.highlight = Highlight[options.highlightScope];
            this.renderer.rows.showTooltip = options.showTooltip;
            this.renderer.rows.showAnnotation = options.showAnnotation;
            this.renderer.rows.sortOnAxisClick = options.sortOnAxisClick;
            this.renderer.columns.sortOnAxisClick = options.sortOnAxisClick;
            this.renderer.render();

            if (options.showSlider) {
                const slider = new Slider(this, options.sliderCaption);
                slider.render();
            }

            this.partitions.adjustScrollableBlocks();

            // if (detached) {
            //     this._container.remove();
            //     return this._container;
            // }

            return this;
        }

        _getMeasureSvg() {
            const svg = d3.select(this._container)
                .append("svg")
                .attr("width", 0)
                .attr("height", 0)
                .style("position", "absolute")
                .style("visibility", "hidden");
            svg.append("text");
            return svg;
        }
    }

    class BubbleMatrixOptions {
        constructor() {
            this.numberIsPercentage = false;
            this.preserveRowOrder = false;
            this.preserveColumnOrder = false;
            this.showSlider = true;
            this.sliderCaption = "Value";
            this.highlightScope = "matrix"; // "matrix", "byRow", "topBottom"
            this.numberOfTopBottom = 5;
            this.showTooltip = true;
            this.showAnnotation = true;
            this.sortOnAxisClick = true;
        }
    }

    class ChartData {
        constructor(dataset, fieldNames) {
            this._dataset = dataset;
            this._fieldNames = fieldNames;

            this.numberIsPercentage = false;
            this.preserveColumnOrder = false;
            this.preserveRowOrder = false;
            this.columns = null;
            this.rows = null

            this.level = 0;
            this.defaultLevel = 0;
            this.average = 0;
            this.min = 0;
            this.max = 0;
            this.numOfTopBottom = 5;
        }

        resetColumns(exception) {
            this.columns.forEach(c => {
                if (c !== exception) c.order = SortOrder.none;
            });
        }

        resetRows(exception) {
            this.rows.forEach(r => {
                if (r !== exception) r.order = SortOrder.none;
            });
        }

        process() {
            const names = this._fieldNames;
            const rows = d3.group(this._dataset, d => d[names.row]);
            this.columns = [...new Set(this._dataset.map(d => d[names.column]))]
                .map((d, i) => new Column(d, i));
            if (!this.preserveColumnOrder) {
                this.columns.sort((a, b) => a.name.localeCompare(b.name));
                this.columns.forEach((c, i) => c.position = i);
            }

            const keys = [...rows.keys()];
            if (!this.preserveRowOrder) keys.sort((a, b) => a.localeCompare(b));
            this.rows = keys.map((key, i) => {
                const srcRow = rows.get(key);
                const cells = this.columns.map(col => {
                    const c = srcRow.find(d => d[names.column] === col.name);
                    return new Cell(key, col.name, c ? +c[names.value] : null, c[names.detail]);
                });
                const row = new Row(key, i, cells);
                row.markBounds();
                return row;
            });

            const cells = this.rows
                .flatMap(r => r.cells)
                .filter(c => c.value)
                .sort((a, b) => a.value - b.value);
            cells[0].flag |= CellFlag.min;
            cells[cells.length - 1].flag |= CellFlag.max;
            cells.slice(0, this.numOfTopBottom).forEach(cell => cell.flag |= CellFlag.bottomGroup);
            cells.slice(-this.numOfTopBottom).forEach(cell => cell.flag |= CellFlag.topGroup);

            const total = cells.reduce((a, b) => a + b.value, 0);
            this.level = this.defaultLevel = this.average = total / cells.length;
            this.min = cells[0].value;
            this.max = cells[cells.length - 1].value;
        }
    }

    class Column {
        constructor(name, position) {
            this.name = name;
            this.position = position;
            this.order = SortOrder.none;
        }
    }

    class Row {
        constructor(name, position, cells) {
            this.name = name;
            this.position = position;
            this.order = SortOrder.none;
            this.cells = cells;
        }

        markBounds() {
            const sorted = [...this.cells].filter(c => c.value).sort((a, b) => a.value - b.value);
            sorted[0].flag |= CellFlag.rowMin;
            sorted[sorted.length - 1].flag |= CellFlag.rowMax;
        }
    }

    class Cell {
        constructor(row, column, value, detail) {
            this.row = row;
            this.column = column;
            this.value = value;
            this.detail = detail;
            this.flag = CellFlag.unspecified;
        }
    }

    class FieldNames {
        constructor(column, row, value, detail) {
            this.column = column;
            this.row = row;
            this.value = value;
            this.detail = detail;
        }
    }

    class SortOrder {
        static get none() { return 0; }
        static get ascending() { return 1; }
        static get descending() { return 2; }
    }

    class CellFlag {
        static get unspecified() { return 0; }
        static get min() { return 1; }
        static get max() { return 2; }
        static get rowMin() { return 4; }
        static get rowMax() { return 8; }
        static get bottomGroup() { return 16; }
        static get topGroup() { return 32; }
    }

    class InfoBox {
        constructor(svg, font, fill, opacity, stroke) {
            this._svg = svg;
            this._font = font;
            this._charBox = null;
            this._box = this._initBox(fill, opacity, stroke);
            this.left = 0;
            this.top = 0;

            this.getBBox = null;
            this.calcTextWidth = null;
            this.calcPosition = null;
        }

        get box() { return this._box; }
        get offset() { return 10; }

        show(e, content, x, y) {
            if (!this._charBox) this._charBox = this.getBBox("M");

            const
                that = this,
                space = 1.4,
                width = this._calcWidth(content);

            this._box
                .style("visibility", "visible")
                .select("rect")
                .attr("width", width + 10)
                .attr("height", `${content.length * space + 0.5}em`);

            drawTexts("backtext", "white", 3);
            drawTexts("foretext");

            this.move(e, x, y);

            function drawTexts(className, stroke, strokeWidth) {
                that._box
                    .selectAll("." + className)
                    .data(content)
                    .join(
                        enter => {
                            enter.append("text")
                                .attr("class", className)
                                .attr("dy", (d, i) => `${space * i + 1}em`)
                                .attr("stroke", stroke)
                                .attr("stroke-width", strokeWidth)
                                .text(d => d);
                        },
                        update => update.text(d => d),
                        exit => exit.remove()
                    );
            }
        }

        // wrap(text, wrapWidth, yAxisAdjustment = 0) {
        //   text.each(function() {
        //     var text = d3.select(this),
        //         words = text.text().split(/\s+/).reverse(),
        //         word,
        //         line = [],
        //         lineNumber = 0,
        //         lineHeight = 1.1, // ems
        //         y = text.attr("y"),
        //         dy = parseFloat(text.attr("dy")) - yAxisAdjustment,
        //         tspan = text.text(null).append("tspan").attr("x", 0).attr("y", y).attr("dy", `${dy}em`);
        //     while (word = words.pop()) {
        //       line.push(word);
        //       tspan.text(line.join(" "));
        //       if (tspan.node().getComputedTextLength() > wrapWidth) {
        //         line.pop();
        //         tspan.text(line.join(" "));
        //         line = [word];
        //         tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
        //       }
        //     }
        //   });
        //   return 0;
        // }


        move(e, x, y) {
            if (this._box) {
                const
                    converted = x && y ? new DOMPoint(x, y) : this._convertCoordinate(e, this._svg),
                    box = this._box.node().getBBox();
                const { left, top } = this.calcPosition(converted, box);
                this.left = left + this.offset;
                this.top = top + this.offset;
                this._box.attr("transform", `translate(${this.left},${this.top})`);
            }
        }

        hide() {
            if (this._box) this._box.style("visibility", "hidden");
        }

        _initBox(fill, opacity, stroke) {
            const box = this._svg
                .append("g")
                .attr("fill", "black")
                .call(g => {
                    g.append("rect")
                        .attr("opacity", opacity)
                        .attr("stroke", stroke)
                        .attr("stroke-width", 0.5)
                        .attr("rx", 4).attr("ry", 4)
                        .attr("x", -5).attr("y", -5)
                        .attr("fill", fill);
                });
            this._font.applyTo(box);
            return box;
        }

        _calcWidth(strs) {
            let max = 0;
            strs.forEach(s => {
                const len = this.calcTextWidth(s);
                if (len > max) max = len;
            });
            return max;
        }

        _convertCoordinate(e, g) {
            const p = this._svg.node().createSVGPoint()
            p.x = e.clientX;
            p.y = e.clientY;
            return p.matrixTransform(g.node().getScreenCTM().inverse());
        }
    }

    class Annotation extends InfoBox {
        constructor(svg, font, fill, opacity, stroke) {
            super(svg, font, fill, opacity, stroke);
            this._pointer = null;
        }

        show(e, content, x, y, r, color) {
            super.show(e, content, x, y);
            const b = this.box.node().getBBox();

            this.move(e, x + r - this.offset, y + r - this.offset);

            const shift = this.offset / 2;
            let
                left = this.left, top = this.top,
                tx = left - shift, ty = top - shift,
                sx = tx + 30, sy = ty + b.height;
            if (left < x) {
                tx = x - r - shift;
                sx = tx - 30;
                left = x - r - b.width;
            }
            if (top < y) {
                ty = y - r - shift;
                sy = ty - b.height;
                top = y - r - b.height;
            }
            this.box.attr("transform", `translate(${left},${top})`);

            this._removePointer();
            this._pointer = this._svg.append("path")
                .attr("fill", "none")
                .attr("stroke", color)
                .attr("stroke-width", 2)
                .attr("d", `M ${x} ${y} L ${tx} ${ty} L ${tx} ${sy} L ${sx} ${sy}`);
        }

        hide() {
            super.hide();
            this._removePointer();
        }

        _removePointer() {
            if (this._pointer) {
                this._pointer.remove();
                this._pointer = null;
            }
        }
    }

    class Measures {
        constructor(svg) {
            this._svg = svg;
            this.font = new Font();

            this.padding = 10;
            this.margin = {
                left: 0,
                right: 0,
                top: 0,
                bottom: 0
            };

            this.width = 0;
            this.height = 0;

            this.sliderHeight = 20;
            this.columnHeight = 0;
            this.rowWidth = 0;

            this._minRadius = 25;
            this.bubbleRadius = 0;
            this.bubbleDiameter = 0;
        }

        initialize(chartData, showSlider) {
            this._calculateLabels(chartData);
            this._calculateBubbleRadius(chartData, showSlider);
        }

        _calculateLabels(chartData) {
            let prefix = "";
            const max = (strs, font) => {
                let max = 0;
                strs.forEach(str => {
                    str = prefix.concat(str);
                    const w = this.calculateStringWidth(str, undefined, font);
                    if (w > max) max = w;
                });
                return max;
            }

            this.columnHeight = rowLabelHeight;  // Workaround
            this.rowWidth = rowLabelWidth;  // Workaround
            // if (chartData.columns.length > 0) {
            //     this.columnHeight = max(chartData.columns.map(c => this.trim(c.name) + "M"));
            // }
            // if (chartData.rows.length > 0) {
            //     this.rowWidth = this.margin.left + max(chartData.rows.map(r => r.name + "MM"), this.font.clone().weight("bold"));
            // }
        }

        _calculateBubbleRadius(chartData, showSlider) {
            const
                aw = this.width - this.rowWidth,
                ah = this.height - this.columnHeight - this.padding * 2 - (showSlider ? this.sliderHeight : 0),
                rc = chartData.rows.length,
                cc = chartData.columns.length;

            const
                r1 = (ah / rc - this.padding) / 2,
                r2 = aw / cc / 2;

            if (r1 < r2) {
                this.bubbleRadius = r1;
            }
            else {
                const total = rc * r2;
                this.bubbleRadius = r2 > ah ? r2 - (total - ah) / rc / 2 : r2;
            }

            if (this.bubbleRadius < this._minRadius) this.bubbleRadius = this._minRadius;
            this.bubbleDiameter = this.bubbleRadius * 2;
        }

        getBBox(str, angle, font) {
            const f = font ?? this.font;
            const text = this._svg.select("text");
            if (text) {
                f.applyTo(text);
                text.text(str);
                if (angle) text.attr("transform", `rotate(${angle})`);
                return text.node().getBBox();
            }
            else {
                return null;
            }
        }

        calculateStringWidth(str, angle, font) {
            const b = this.getBBox(str, angle, font);
            return b
                ? Math.sqrt(b.width * b.width + b.height * b.height)
                : str.length * this.font.size;
        }

        trim(s) {
            return s.length > 100 ? `${s.substr(0, 10)}...` : s;
        }
    }

    class Font {
        constructor(family = "", size = "10pt", style = "normal", weight = "normal") {
            this._family = family;
            this._size = size;
            this._style = style;
            this._weight = weight;
        }

        family(_) { return arguments.length ? (this._family = _, this) : this._family; }
        size(_) { return arguments.length ? (this._size = _, this) : this._size; }
        style(_) { return arguments.length ? (this._style = _, this) : this._style; }
        weight(_) { return arguments.length ? (this._weight = _, this) : this._weight; }


        applyTo(elem) {
            elem = elem instanceof HTMLElement ? d3.select(elem) : elem;
            elem.style("font-family", this._family)
                .style("font-size", isNaN(+this._size) ? this._size : `${this._size}pt`)
                .style("font-style", this._style)
                .style("font-weight", this._weight);
        }

        clone() {
            return new Font(this._family, this._size, this._style, this._weight);
        }
    }

    class Partitions {
        constructor(chart) {
            this._chart = chart;

            this.chartArea = this._createDiv();
            this.mainBlock = this._createDiv();
            this.slider = this._createDiv();
            this.columnBlock = this._createDiv();
            this.placeHolder = this._createDiv();
            this.columns = this._createDiv();
            this.matrixBlock = this._createDiv();
            this.rows = this._createDiv();
            this.matrix = this._createDiv();
        }

        initialize() {
            this._initStyles();
            this._initLayout();
            this._adjustSize();
        }

        adjustScrollableBlocks() {
            const
                gh = this.matrix.offsetHeight - this.matrix.clientHeight,
                gw = this.matrix.offsetWidth - this.matrix.clientWidth;
            if (gh) this.rows.style.height = `${this.rows.clientHeight - gh}px`;
            if (gw) this.columns.style.width = `${this.columns.clientWidth - gw}px`;
        }

        changeFont(font) {
            this.chartArea.style.font = font.shorthand;
        }

        _adjustSize() {
            const measures = this._chart.measures;
            this.chartArea.style.height = `${measures.height}px`;
            if (this._chart.options().showSlider) this.slider.style.height = `${measures.sliderHeight}px`;
            this.columns.style.height = `${measures.columnHeight}px`;
            this.columns.style.width = `${measures.width - measures.rowWidth - measures.margin.left}px`;
            // console.log("Partitions adjustSize width", measures.width, measures.rowWidth, measures.margin.left, this.columns.style.width);  // TEMP
            this.placeHolder.style.width = `${measures.rowWidth + measures.margin.left}px`;
            this.placeHolder.style.height = `${measures.columnHeight}px`;
            this.rows.style.width = `${measures.rowWidth + measures.margin.left}px`;
            this.matrix.style.width = `${measures.width + measures.margin.right + measures.rowWidth}px`;
        }

        _createDiv() {
            return document.createElement("div");
        }

        _initStyles() {
            const
                measures = this._chart.measures,
                width = measures.width + measures.margin.left + measures.margin.right;

            // chartArea
            let s = this.chartArea.style;
            s.display = "flex";
            s.flexDirection = "column";
            s.cursor = "default";
            s.userSelect = "none";
            measures.font.applyTo(this.chartArea);

            // mainBlock
            s = this.mainBlock.style;
            s.display = "flex";
            s.flexDirection = "column";
            s.flexGrow = 1;
            s.height = "1px";

            // columnBlock
            s = this.columnBlock.style;
            s.textAlign = "left";
            s.display = "flex";
            s.flexDirection = "row";
            s.flexShrink = 0;
            s.width = "1px";

            // placeHolder
            s = this.placeHolder.style;
            s.flexShrink = 0;
            s.backgroundColor = "white";

            // columns
            s = this.columns.style;
            s.paddingTop = "5px";
            s.paddingBottom = "5px";
            s.overflow = "hidden";
            s.flexShrink = 0;
            s.width = `${width}px`;

            // matrixBlock
            s = this.matrixBlock.style;
            s.display = "flex";
            s.flexDirection = "row";
            s.flexGrow = 0;
            s.width = "1px";
            s.overflowY = "auto";
            s.textAlign = "left";
            s.width = `${width}px`;

            // rows
            s = this.rows.style;
            s.overflow = "hidden";
            s.flexShrink = 0;

            // matrix
            s = this.matrix.style;
            s.overflowX = "auto";
            this.matrix.onscroll = ev => {
                this.columns.scrollLeft = this.matrix.scrollLeft;
                this.rows.scrollTop = this.matrix.scrollTop;
            }
        }

        _initLayout() {
            this.chartArea.appendChild(this.mainBlock);
            this.mainBlock.appendChild(this.slider);
            this.mainBlock.appendChild(this.columnBlock);
            this.mainBlock.appendChild(this.matrixBlock);
            this.columnBlock.appendChild(this.placeHolder);
            this.columnBlock.appendChild(this.columns);
            this.matrixBlock.appendChild(this.rows);
            this.matrixBlock.appendChild(this.matrix);
        }
    }

    class Renderer {
        constructor(chart) {
            this.chart = chart;

            this.colors = {
                row: "#caf0f8",
                above: "#ffd166",
                below: "#118ab2",
                label: "#3c3c43"
            };

            this.matrix = {
                svg: null,
                g: null,
                og: null,
                ig: null
            };

            this.columns = new ColumnRenderer(this);
            this.rows = new RowRenderer(this);

            this._uuid = `uu${Date.now()}${Math.floor(Math.random() * 10000)}`;
        }

        render() {
            this._initMatrix();
            this._renderGradients();
            this.columns.render();
            this.rows.render();
        }

        _createSvg(container, width, height) {
            return d3.select(container)
                .append("svg")
                .attr("width", width + 100)
                .attr("height", height);
        }

        _transform(x, y) {
            return `translate(${x},${y})`;
        }

        _initMatrix() {
            const
                c = this.chart,
                m = this.matrix;
            m.svg = this._createSvg(c.partitions.matrix, c.scales.maxX, c.scales.maxY);
            m.g = m.svg.append("g");
            m.og = m.g.append("g");
            m.ig = m.svg.append("g").attr("transform", this._transform(0, c.measures.padding / 2));
        }

        uid(id) {
            return `${this._uuid}_${id}`;
        }

        _renderGradients() {
            const
                c = this.chart,
                addGradient = reversed => {
                    this.matrix.g.append("linearGradient")
                        .attr("id", this.uid(reversed ? "descending" : "ascending"))
                        .attr("x1", 0)
                        .attr("x2", c.measures.rowWidth + c.chartData.columns.length * c.measures.bubbleDiameter)
                        .attr("y1", "100%")
                        .attr("y2", "100%")
                        .attr("gradientUnits", "userSpaceOnUse")
                        .call(g => {
                            g.append("stop").attr("stop-color", this.colors.row).attr("stop-opacity", reversed ? 1 : 0).attr("offset", 0);
                            g.append("stop").attr("stop-color", this.colors.row).attr("stop-opacity", reversed ? 0 : 1).attr("offset", 1);
                        });
                };

            addGradient(false);
            addGradient(true);
        }
    }

    class PartRenderer {
        constructor(mainRenderer) {
            this._mainRenderer = mainRenderer;
            this.duration = 500;
        }

        get chart() { return this._mainRenderer.chart; }
        get matrix() { return this._mainRenderer.matrix; }
        get colors() { return this._mainRenderer.colors; }
        get rows() { return this._mainRenderer.rows; }
        get columns() { return this._mainRenderer.columns; }

        uid(id) {
            return this._mainRenderer.uid(id);
        }

        url(id) {
            return `url(#${id})`;
        }
    }

    class ColumnRenderer extends PartRenderer {
        constructor(mainRenderer) {
            super(mainRenderer);
            this.text = null;
            this._arrow = null;
            this.labels = null;
            this.axis = null;
            this._focused = null;
            this.sortOnAxisClick = true;
        }

        render() {
            const c = this.chart;
            // console.log("ColumnRenderer width", c.scales.maxX, c.measures.rowWidth); // TEMP
            const bufferWidth = 0;
            const bufferHeight = 0;
            const g = d3.select(c.partitions.columns)
                .append("svg")
                .attr("width", c.scales.maxX + c.measures.rowWidth + bufferWidth)
                .attr("height", c.measures.columnHeight + bufferHeight)
                .append("g");
            
            this.labels = this._renderGroups(
                g,
                `translate(0,${c.measures.columnHeight})`,
                "start",
                g => {
                    this.text = g.append("text")
                        .attr("y", 0)
                        .attr("dy", "-0.25em")
                        .attr("transform", "rotate(-45)")
                        .attr("fill", this.colors.label)
                        .text(this._trim.bind(this))
                        .on("click", this._click.bind(this))
                        .on("pointerenter", this._handlePointerEnter.bind(this))
                        .on("pointerleave", this._handlePointerLeave.bind(this));

                    // this._arrow = g.append("path")
                    //   .attr("fill", "#999")
                    //   .attr("transform", "rotate(-45)");
                }
            );
            
            const axisColor = "#a7a7a7";
            this.axis = this._renderGroups(
                this.matrix.og,
                undefined,
                "middle",
                g => {
                    g.append("line")
                        .attr("y1", 0).attr("y2", c.scales.maxY)
                        .attr("stroke-width", 1).attr("stroke", axisColor);
                }
            );
        }

        _renderGroups(g, transform, textAnchor, draw) {
            const
                c = this.chart,
                gg = g.append("g");
            if (transform) gg.attr("transform", transform);
            return gg.selectAll("g")
                .data(c.chartData.columns)
                .join("g")
                .attr("text-anchor", textAnchor)
                .attr("transform", (d, i) => `translate(${c.scales.x(i) + c.measures.bubbleRadius},0)`)
                .call(draw);
        }

        _trim(d) {
            const prefix = "";
            return this.chart.measures.trim(prefix + (d.name));
        }

        _click(e, d) {
            this._sort(d);
            if (this._arrow) {
                this._arrow.attr("d", c => {
                    if (d.order !== SortOrder.none && d.name === c.name) {
                        const
                            b = this.chart.measures.getBBox(this._trim(c)),
                            h = b.height / 2 + 1, w = b.width, aw = 5;
                        return d.order === SortOrder.descending
                            ? `M 0 ${h} L ${w} ${h} L ${w} ${h + aw} L 0 ${h}`
                            : `M 0 ${h} L 0 ${h + aw} L ${w} ${h} L 0 ${h}`;
                    }
                })
            }

            if (this._focused !== d) {
                this.columns.axis.select("line").attr("stroke-width", col => col.name === d.name ? 2 : 1);
                this.columns.text.attr("font-weight", col => col.name === d.name ? "bold" : "");
                this._focused = d;
                dispatch("message", {selection_type: "col", col: ("" + d.name), row: null, sortOrder: d.order})
                if (this.rows._rowFocused != null) {
                    this.rows._bubbleRects.attr("stroke-width", 0);
                    this.rows._labelRects.attr("stroke-width", 0);
                    this.rows._rowFocused = null;
                }
            } 
            else {
                this.columns.axis.select("line").attr("stroke-width", 1);
                this.columns.text.attr("font-weight", "");
                this._focused = null;
                dispatch("message", {selection_type: "col", col: null, row: null, sortOrder: d.order})
            }
        }

        _sort(d) {
            if (this.sortOnAxisClick) {
                const
                    that = this,
                    data = this.chart.chartData,
                    columns = data.columns,
                    cIndex = columns.indexOf(d),
                    sorted = data.rows.map(d => ({ row: d, column: d.cells[cIndex] }));

                data.resetColumns(d);
                if (d.order === SortOrder.none) d.order = SortOrder.descending;
                else if (d.order === SortOrder.descending) d.order = SortOrder.ascending;
                else d.order = SortOrder.none;

                if (d.order === SortOrder.ascending) {
                    sorted.sort((a, b) => a.column.value - b.column.value);
                }
                else if (d.order === SortOrder.descending) {
                    sorted.sort((a, b) => b.column.value - a.column.value);
                }

                const unit = this.duration / data.rows.length;
                sortRows(this.rows.axis);
                sortRows(this.rows.labels);
                this.rows.relocateAnnotation();

                function sortRows(g) {
                    g.transition()
                        .duration((d, i) => i * unit)
                        .attr("transform", d => {
                            let rIndex = 0;
                            for (let i = 0; i < sorted.length; i++) {
                                if (sorted[i].row === d) {
                                    d.position = rIndex = i;
                                    break;
                                }
                            }
                            return `translate(0,${that.chart.scales.y(rIndex)})`;
                        });
                }
            }
        }

        _handlePointerEnter(e, d) {
            if (this.onhover) this.onhover(e, d);
                this.axis.select("line").attr("stroke-width", col => ((col.name === d.name) || (this._focused != null && col.name === this._focused.name)) ? 2 : 1);
                this.text.attr("font-weight", col => ((col.name === d.name) || (this._focused != null && col.name === this._focused.name)) ? "bold" : "");
        }

        _handlePointerLeave(e, d) {
            if (this._focused == null) {
                if (this.oncancel) this.oncancel(e, d);
                    this.axis.select("line").attr("stroke-width", 1);
                    this.text.attr("font-weight", "");
            } else {
                this.axis.select("line").attr("stroke-width", col => col.name === this._focused.name ? 2 : 1);
                this.text.attr("font-weight", col => col.name === this._focused.name ? "bold" : "");
            }
        }
    }

    class RowRenderer extends PartRenderer {
        constructor(mainRenderer) {
            super(mainRenderer);
            this._tooltip = null;
            this._annotation = null;
            this.labels = null;
            this.axis = null;
            this.highlight = Highlight.matrix;
            this.showTooltip = true;
            this.showAnnotation = true;
            this.sortOnAxisClick = true;

            this._labelRects = null;
            this._bubbleRects = null;
            this._bubbles = null;
            this._focused = null;
            this._rowFocused = null;

            this.onhover = null;
            this.oncancel = null;
            this.onclick = null;
        }

        get bubbles() { return this._bubbles; }

        render() {
            this._initInfoLayer();

            const c = this.chart;
            const g = d3.select(c.partitions.rows)
                .append("svg")
                .attr("width", c.measures.rowWidth)
                .attr("height", c.scales.maxY)
                .append("g")
                .attr("transform", `translate(0,${c.measures.padding / 2})`);

            this.labels = this._renderGroups(g, g => {
                this._labelRects = this._renderRect(g, 1, c.measures.rowWidth + 10)
                    .on("click", this._click.bind(this));

                g.append("text")
                    .attr("font-weight", "bold")
                    .attr("y", c.measures.bubbleRadius)
                    .attr("dx", "1em")
                    .attr("dy", "0.25em")
                    .attr("fill", this.colors.label)
                    .text(d => d.name)
                    .on("click", this._click.bind(this));
            });

            const rectWidth = c.chartData.columns.length * c.measures.bubbleDiameter;
            this.axis = this._renderGroups(this.matrix.ig.append("g"), g => {
                this._bubbleRects = this._renderRect(g, -11, rectWidth + 10);
                this._bubbles = g.append("g")
                    .selectAll("g")
                    .data(d => d.cells)
                    .join("g")
                    .attr("transform", (d, i) => `translate(${c.scales.x(i) + c.measures.bubbleRadius},0)`)
                    .call(g => {
                        g.append("circle")
                            .attr("class", "bubble")
                            .attr("fill", d => d.value >= c.chartData.average ? this.colors.above : this.colors.below)
                            .attr("opacity", 0.5)
                            .attr("stroke-width", 2)
                            .attr("cy", c.measures.bubbleRadius)
                            .attr("r", 0);
                    })
                    .on("click", this._handleClick.bind(this))
                    .on("pointerenter", this._handlePointerEnter.bind(this))
                    .on("pointermove", this._handlePointerMove.bind(this))
                    .on("pointerleave", this._handlePointerLeave.bind(this));

                this._bubbles.selectAll("circle")
                    .transition()
                    .ease(d3.easeBounce)
                    .duration(500)
                    .attr("r", d => d.value ? c.scales.r(d.value) : 0);
            });

            // this.highlightBubbles(); // Remove highlighting
        }

        relocateAnnotation(delay = true) {
            if (this._focused) {
                const f = () => this._showAnnotation(null, this._focused);
                this._annotation.hide();
                if (delay) {
                    setTimeout(f, this.duration);
                }
                else {
                    f();
                }

            }
        }

        _handleClick(e, d) {
            const c = this.chart;
            if (this.onclick) this.onclick(e, d);
        }

        _showAnnotation(e, d) {
            const
                c = this.chart,
                a = this._annotation,
                getPosition = (axis, name) => {
                    const obj = axis.find(d => d.name === name);
                    return obj ? obj.position : 0;
                }

            if (this.showAnnotation) {
                const
                    cx = c.scales.x(getPosition(c.chartData.columns, d.column)),
                    cy = c.scales.y(getPosition(c.chartData.rows, d.row)),
                    r = c.scales.r(d.value),
                    color = d.value >= c.chartData.level ? this.colors.above : this.colors.below;

                this._focused = d;
                a.hide();
                a.show(
                    null, this._getTooltipContent(d),
                    cx + c.measures.bubbleRadius,
                    cy + c.measures.bubbleRadius + c.measures.padding / 2,
                    r, d3.color(color).darker(1));
            }
        }

        _handlePointerEnter(e, d) {
            const c = this.chart;

            if (this.onhover) this.onhover(e, d);
                this.columns.axis.select("line").attr("stroke-width", col => col.name === d.column ? 2 : 1);
                // this.columns.text.attr("font-weight", col => col.name === d.column ? "bold" : "");

            if (this._focused == null) {
                // If nothing is selected, filter on hover-enter
                if (this.showTooltip) this._tooltip.show(e, this._getTooltipContent(d));
                if (this.onhover) this.onhover(e, d);
                // dispatch("message", {selection_type: "cell", row: ("" + d.row) , col: ("" + d.column)})

                // Add darker bubble
                this._bubbles.filter(b => b === d)
                .call(g => {
                    g.insert("circle", "circle")
                    .attr("class", "shadow")
                    .attr("cy", c.measures.bubbleRadius)
                    .attr("r", d => d.value ? c.scales.r(d.value) : 0)
                    .attr("fill", d => {
                        const color = d.value >= c.chartData.level ? this.colors.above : this.colors.below;
                        return d3.color(color).darker(1);
                    })
                    .attr("opacity", 0.5);
                });
            }
        }

        _handlePointerMove(e) {
            if (this.showTooltip) this._tooltip.move(e);
        }

        _handlePointerLeave(e, d) {
            if (this.showTooltip) this._tooltip.hide();
            if (this._focused == null) {
                // dispatch("message", {selection_type: "cell", row: null , col: null})

                if (this.oncancel) this.oncancel(e, d);
                    this.columns.axis.select("line").attr("stroke-width", 1);
                    // this.columns.text.attr("font-weight", "");
                    this._bubbles.filter(b => b === d)
                        .call(g => {
                            g.select(".bubble").attr("transform", "");
                            g.select(".shadow").remove();
                        });
            }
            
        }

        highlightBubbles(update) {
            const bubbleRadius = this.chart.measures.bubbleRadius;
            const
                testByRow = d => this.highlight === Highlight.byRow && ((d.flag & CellFlag.rowMin) === CellFlag.rowMin || (d.flag & CellFlag.rowMax) === CellFlag.rowMax),
                testByMatrix = d => this.highlight === Highlight.matrix && ((d.flag & CellFlag.min) === CellFlag.min || (d.flag & CellFlag.max) === CellFlag.max),
                testByTop = d => this.highlight === Highlight.top && ((d.flag & CellFlag.topGroup) === CellFlag.topGroup),
                testByBottom = d => this.highlight === Highlight.bottom && ((d.flag & CellFlag.bottomGroup) === CellFlag.bottomGroup);

            const targets = this._bubbles.filter(d => d.value && (testByRow(d) || testByMatrix(d) || testByTop(d) || testByBottom(d)));
            targets.select("circle")
                .attr("stroke", d => {
                    const color = d.value >= this.chart.chartData.level ? this.colors.above : this.colors.below;
                    return d3.color(color).darker(0.75);
                });

            if (!update) {
                targets.append("text")
                    .attr("text-anchor", "middle")
                    .attr("y", d => {
                        const
                            r = this.chart.scales.r(d.value),
                            tl = this.chart.measures.calculateStringWidth(this._formatValue(d.value));
                        return tl + 5 > r * 2 ? bubbleRadius + r + 12 : bubbleRadius;
                    })
                    .attr("dy", "0.3em")
                    .attr("fill", this.colors.label)
                    .attr("font-weight", "bold")
                    .text(d => this._formatValue(d.value));
            }
        }

        _formatValue(v, short = true) {
            const fmtStr = this.chart.chartData.numberIsPercentage ? ".1%" : short ? ".2s" : ",.2f";
            return d3.format(fmtStr)(v);
        }

        _initInfoLayer() {
            const
                that = this,
                currFont = this.chart.measures.font;

            if (this.showAnnotation) {
                const font = currFont.clone().family("system-ui").size("11px").weight("bold");
                this._annotation = new Annotation(this.matrix.svg, font, "none");
                assignDelegates(this._annotation, font);
            }

            if (this.showTooltip) {
                this._tooltip = new InfoBox(this.matrix.svg, currFont, "white", 0.7, "#aaa");
                assignDelegates(this._tooltip, currFont);
            }

            function assignDelegates(obj, font) {
                obj.getBBox = s => that.chart.measures.getBBox(s, undefined, font);
                obj.calcTextWidth = s => that.chart.measures.calculateStringWidth(s, undefined, font);
                obj.calcPosition = (c, b) => that._calcTooltipPosition(c, b);
            }
        }

        _getTooltipContent(d) { 
            const names = this.chart.fieldNames;
            return [
                // `Concept: ${d.row}`,
                // `Slice: ${d.column}`,
                // `Matches: ${this._formatValue(d.value, false)}`,
                `${this._formatValue(d.value, false)}`,
            ];
        }

        _calcTooltipPosition(c, box) {
            const
                mBox = this.matrix.svg.node().getBoundingClientRect(),
                x = c.x + mBox.left,
                y = c.y + mBox.top;

            const
                t = 5,
                left = x + box.width + t > mBox.right ? c.x - box.width - t : c.x + t,
                top = y + box.height + t > mBox.bottom ? c.y - box.height - t : c.y + t;
            return { left, top };
        }

        _renderGroups(g, draw) {
            const c = this.chart;
            return g.selectAll("g")
                .data(c.chartData.rows)
                .join("g")
                .attr("transform", (d, i) => `translate(0,${c.scales.y(i)})`)
                .call(draw)
                .on("click", this._rowClick.bind(this))
                .on("pointerenter", (e, d) => {
                    // this._bubbleRects.attr("opacity", row => row.name === d.name ? 0.8 : 0.5);
                    // this._labelRects.attr("opacity", row => row.name === d.name ? 0.8 : 0.5);
                    this._bubbleRects.attr("fill", row => row.name === d.name ? d3.color(this.colors.row).darker(0.3) : this.colors.row);
                    this._labelRects.attr("fill", row => row.name === d.name ? d3.color(this.colors.row).darker(0.3) : this.colors.row); 
                    // this._bubbleRects.attr("stroke-width", row => row.name === d.name ? 2 : 0);
                    // this._labelRects.attr("stroke-width", row => row.name === d.name ? 2 : 0);
                })
                .on("pointerleave", (e, d) => {
                    // this._bubbleRects.attr("opacity", 0.5);
                    // this._labelRects.attr("opacity", 0.5);
                    // this._bubbleRects.attr("stroke-width", 0);
                    // this._labelRects.attr("stroke-width", 0);
                    this._bubbleRects.attr("fill", this.colors.row);
                    this._labelRects.attr("fill", this.colors.row);
                });
        }

        _rowClick(e, d) {
            if (this._rowFocused !== d) {
                this._bubbleRects.attr("stroke-width", row => row.name === d.name ? 2 : 0);
                this._labelRects.attr("stroke-width", row => row.name === d.name ? 2 : 0);
                this._bubbleRects.attr("fill", row => row.name === d.name ? d3.color(this.colors.row).darker(0.3) : this.colors.row);
                this._labelRects.attr("fill", row => row.name === d.name ? d3.color(this.colors.row).darker(0.3) : this.colors.row); 
                this._rowFocused = d;
                dispatch("message", {selection_type: "row", row: ("" + d.name), col: null, sortOrder: d.order})
                if (this.columns._focused != null) {
                    this.columns.axis.select("line").attr("stroke-width", 1);
                    this.columns.text.attr("font-weight", "");
                    this.columns._focused = null;
                }
            } else {
                this._bubbleRects.attr("stroke-width", 0);
                this._labelRects.attr("stroke-width", 0);
                this._bubbleRects.attr("fill", this.colors.row);
                this._labelRects.attr("fill", this.colors.row);
                this._rowFocused = null;
                dispatch("message", {selection_type: "row", row: null, col: null, sortOrder: d.order})
            }
        }

        _renderRect(g, x, width) {
            const c = this.chart;
            return g.append("rect")
                .attr("width", width)
                .attr("height", c.measures.bubbleDiameter)
                .attr("x", x)
                .attr("rx", 10)
                .attr("opacity", 0.5)
                .attr("fill", this.colors.row)
                .attr("stroke", d3.color(this.colors.row).darker(1))
                .attr("stroke-width", 0);
        }

        _click(e, d) {
            this._sort(d);
            this._labelRects.attr("fill", d => d.order === SortOrder.ascending ? "white" : this.colors.row);
            this._bubbleRects.attr(
                "fill",
                _ => {
                    if (_.order === SortOrder.none) {
                        return this.colors.row;
                    }
                    else {
                        const gradId = this.uid(_.order === SortOrder.ascending ? "ascending" : "descending");
                        return _ === d ? this.url(gradId) : this.colors.row;
                    }
                }
            )
        }

        _sort(d) {
            if (this.sortOnAxisClick) {
                const
                    that = this,
                    indices = [],
                    data = this.chart.chartData,
                    sorted = d.cells.map(v => v);

                data.resetRows(d);
                if (d.order === SortOrder.none) d.order = SortOrder.descending;
                else if (d.order === SortOrder.descending) d.order = SortOrder.ascending;
                else d.order = SortOrder.none;

                if (d.order === SortOrder.ascending) {
                    sorted.sort((a, b) => a.value - b.value);
                }
                else if (d.order === SortOrder.descending) {
                    sorted.sort((a, b) => b.value - a.value);
                }

                const
                    unit = this.duration / data.columns.length,
                    x = this.chart.scales.x,
                    bubbleRadius = this.chart.measures.bubbleRadius;
                sortColumn(this.columns.axis);
                sortColumn(this.columns.labels);
                this.relocateAnnotation();


                this._bubbles.transition()
                    .duration((d, i) => i * unit)
                    .attr("transform", (b, i) => `translate(${x(indices[i]) + bubbleRadius},0)`);

                function sortColumn(g) {
                    g.transition()
                        .duration((d, i) => i * unit)
                        .attr("transform", c => {
                            let cIndex = 0;
                            for (let i = 0; i < sorted.length; i++) {
                                if (sorted[i].column === c.name) {
                                    c.position = cIndex = i;
                                    indices.push(i);
                                    break;
                                }
                            }
                            return `translate(${x(cIndex) + bubbleRadius},0)`;
                        });
                }
            }
        }
    }

    class Highlight {
        static get matrix() { return 0; }
        static get byRow() { return 1; }
        static get top() { return 2; }
        static get bottom() { return 3; }
    }

    class Scales {
        constructor() {
            this.x = null;
            this.y = null;
            this.r = null;
            this._maxX = 0;
            this._maxY = 0;
        }

        get maxX() { return this._maxX; }
        get maxY() { return this._maxY; }

        initialize(chart) {
            const measures = chart.measures;
            const chartData = chart.chartData;

            this.x = i => i * measures.bubbleDiameter;
            this.y = i => i * (measures.bubbleDiameter + measures.padding);
            this.r = d3.scaleLinear()
                .domain([chartData.min, chartData.max])
                .range([measures.bubbleRadius * 0.2, measures.bubbleRadius - 1.5]);
            this._maxX = this.x(chartData.columns.length);
            this._maxY = this.y(chartData.rows.length);
        }
    }

    class Slider {
        constructor(chart, caption) {
            this._chart = chart;
            this._caption = caption;

            this._g = null;
            this._label = null;
            this._below = null;
            this._above = null;

            this._width = 0;
            this._min = 0;
            this._max = 0;
            this._defaultValue = 0;
        }

        get level() { return this._chart.chartData.level; }
        set level(v) { this._chart.chartData.level = v; }
        get defaultLevel() { return this._chart.chartData.defaultLevel; }
        get isPercent() { return this._chart.chartData.numberIsPercentage; }
        get height() { return this._chart.measures.sliderHeight; }
        get rowRenderer() { return this._chart.renderer.rows; }

        render() {
            const measures = this._chart.measures;

            this._initialize();
            this._g = d3.select(this._chart.partitions.slider)
                .append("svg")
                .attr("width", measures.width)
                .attr("height", this.height)
                .append("g");

            this._renderLabel();
            this._renderColorRects();
            this._renderSlider();
        }

        _initialize() {
            const
                measures = this._chart.measures,
                chartData = this._chart.chartData;

            this._defaultValue = this.isPercent ? this.defaultLevel * 100 : this._roundUp(this.defaultLevel, 1),
                this._width = (measures.width - measures.rowWidth) / 2;

            let min = chartData.min, max = chartData.max;
            if (this.isPercent) {
                min = min * 100;
                max = max * 100 + 1;
                min = min > 0 ? min - 1 : min;
            }
            this._min = min;
            this._max = max;
        }

        _renderLabel() {
            this._label = this._g.append("text")
                .attr("x", this._width + 12)
                .attr("y", 10)
                .attr("dy", "0.5em")
                .attr("fill", "black");
            this._updateLabel(this._defaultValue);
        }

        _renderColorRects() {
            const { a, b } = this._getSafeBound();
            const x = (this._defaultValue - a) / (b - a) * this._width;

            this._below = this._g.append("rect")
                .attr("y", 2)
                .attr("rx", 5)
                .attr("width", x).attr("height", this.height - 2)
                .attr("opacity", 0.5)
                .attr("fill", this._chart.renderer.colors.below);

            this._above = this._g.append("rect")
                .attr("x", x).attr("y", 2)
                .attr("rx", 5)
                .attr("width", this._width - x)
                .attr("height", this.height - 2)
                .attr("opacity", 0.5)
                .attr("fill", this._chart.renderer.colors.above);
        }

        _renderSlider() {
            const { a, b } = this._getSafeBound();

            const fo = this._g.append("foreignObject")
                .attr("width", this._width + 2)
                .attr("height", this.height);

            this._slider = fo.append("xhtml:input")
                .attr("type", "range")
                .attr("min", a).attr("max", b)
                .style("width", `${this._width - 5}px`)
                .style("height", `${this._height}px`)
                .on("click", e => e.stopPropagation())
                .on("dblclick", e => {
                    this._slider.node().value = this._defaultValue;
                    this._change();
                    e.stopPropagation();
                })
                .on("input", () => this._change());
            this._slider.node().value = this._defaultValue;
        }

        _change() {
            const { a, b } = this._getSafeBound();
            const
                v = +this._slider.node().value,
                vv = v < a ? a : v > b ? b : v,
                x = (vv - a) / (b - a) * this._width;

            this._below.attr("width", x);
            this._above.attr("x", x).attr("width", this._width - x);
            this._updateLabel(v);
            this.level = this.isPercent ? v / 100 : v;

            const colors = this._chart.renderer.colors;
            this.rowRenderer.bubbles
                .selectAll("circle")
                .transition()
                .duration(1000)
                .ease(d3.easeBounce)
                .attr("fill", d => d.value >= this.level ? colors.above : colors.below)
                .attr("opacity", 0.5);
            // this.rowRenderer.highlightBubbles(true); // TEMP: disable highlighting
            this.rowRenderer.relocateAnnotation(false);
        }

        _getSafeBound() {
            const
                a = this.isPercent ? this._min : this._roundDown(this._min, 2),
                b = this.isPercent ? this._max : this._roundUp(this._max, 2);
            return { a, b };
        }

        _roundUp(n, precision) {
            const
                f = n < 0 ? -1 : 1,
                s = Math.ceil(Math.abs(n)).toString(),
                d = Math.pow(10, s.length - precision);
            return Math.ceil(+s / d) * d * f;
        }

        _roundDown(n, precision) {
            const
                f = n < 0 ? -1 : 1,
                g = n < 0 ? 1 : 0,
                s = Math.floor(Math.abs(n)).toString(),
                d = Math.pow(10, s.length - precision);
            return (Math.floor(+s / d) + g) * d * f;
        }

        _updateLabel(value) {
            const v = value.toFixed(0);
            const vStr = this.isPercent ? `${v}%` : d3.format(",.2r")(v);
            this._label.text(`${this._caption} = ${vStr}`);
        }
    }

</script>

<div>
</div>

<style>
</style>
