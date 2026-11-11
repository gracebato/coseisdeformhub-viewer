// Single-layer downloads from a COSEIS netCDF, converted to a Cloud-Optimized GeoTIFF in the browser.
//
// A product's layers.json (ops/viewer/build_layer_index.py in cmspeed/coseis-sar) says where each layer's bytes are in the
// hosted netCDF: one range for a contiguous layer, or the range of every HDF5 chunk. readLayer()
// fetches only those bytes with HTTP range requests and undoes the HDF5 filters (deflate,
// shuffle); encodeCOG() writes a tiled, DEFLATE-compressed GeoTIFF with internal overviews in the
// COG layout (IFDs first, smallest overview's tiles first). A 3-D layer (height x latitude x longitude:
// imaging geometry, solid earth tides) becomes one band per height, as GDAL reads it. No server-side
// conversion and no second copy of the data: only the netCDF is hosted.
//
// Plain ES module with no dependencies (CompressionStream, DecompressionStream, fetch and
// crypto.subtle are built into current browsers and Node >= 18), so the same code is tested in Node.

const DTYPES = {
    // numpy dtype: [typed array, TIFF SampleFormat, bits]
    '<f4': [Float32Array, 3, 32],
    '<f8': [Float64Array, 3, 64],
    '|u1': [Uint8Array, 1, 8],
    '|i1': [Int8Array, 2, 8],
    '<u2': [Uint16Array, 1, 16],
    '<i2': [Int16Array, 2, 16],
    '<u4': [Uint32Array, 1, 32],
    '<i4': [Int32Array, 2, 32],
};
const PART_BYTES = 8 * 1024 * 1024;  // contiguous layers are fetched in parts of this size
const TILE = 512;

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new Error('layer_download.js assumes a little-endian platform');
}

export class StaleIndexError extends Error {}

async function fetchRange(url, offset, length, signal) {
    const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` }, signal });
    if (response.status !== 206) {
        if (response.body) response.body.cancel();  // a 200 would be the whole file
        throw new Error(`the server answered ${response.status} instead of a partial response`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== length) throw new Error(`expected ${length} bytes, got ${bytes.length}`);
    return bytes;
}

async function transform(bytes, stream) {
    const out = new Blob([bytes]).stream().pipeThrough(stream);
    return new Uint8Array(await new Response(out).arrayBuffer());
}

const inflate = bytes => transform(bytes, new DecompressionStream('deflate'));  // zlib format, as HDF5 and TIFF use
const deflate = bytes => transform(bytes, new CompressionStream('deflate'));

function unshuffle(bytes, size) {
    const n = bytes.length / size;
    const out = new Uint8Array(bytes.length);
    for (let b = 0; b < size; b++) {
        const src = b * n;
        for (let i = 0; i < n; i++) out[i * size + b] = bytes[src + i];
    }
    return out;
}

async function pool(tasks, limit) {
    let next = 0;
    const run = async () => {
        while (next < tasks.length) await tasks[next++]();
    };
    await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, run));
}

function nodata(layer) {
    return layer.fill_is_nan ? NaN : layer.fill;
}

// Checks the layer's first stored bytes against the index; throws StaleIndexError if the file changed.
export async function verifyProbe(url, layer, signal) {
    const { offset, length, sha256 } = layer.probe;
    const digest = await crypto.subtle.digest('SHA-256', await fetchRange(url, offset, length, signal));
    const hex = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    if (hex !== sha256) throw new StaleIndexError('the hosted file no longer matches its layer index');
}

// [bands, rows, columns] of a 2-D or 3-D layer.
function dims(layer) {
    return layer.shape.length === 3 ? layer.shape : [1, ...layer.shape];
}

// Fetches one layer; returns a typed array of all its values, row-major (band after band for a 3-D layer).
export async function readLayer(url, layer, { concurrency = 6, onProgress = () => {}, signal } = {}) {
    const [Typed] = DTYPES[layer.dtype] || [];
    if (!Typed) throw new Error(`unsupported type ${layer.dtype}`);
    const [bands, height, width] = dims(layer);
    const size = Typed.BYTES_PER_ELEMENT;
    const out = new Typed(bands * height * width);
    let done = 0;
    const progress = n => onProgress((done += n) / layer.bytes);
    await verifyProbe(url, layer, signal);

    if (layer.storage === 'contiguous') {
        const target = new Uint8Array(out.buffer);
        if (layer.bytes !== target.length) throw new Error('contiguous layer size does not match its shape');
        const tasks = [];
        for (let start = 0; start < layer.bytes; start += PART_BYTES) {
            const length = Math.min(PART_BYTES, layer.bytes - start);
            tasks.push(async () => {
                target.set(await fetchRange(url, layer.offset + start, length, signal), start);
                progress(length);
            });
        }
        await pool(tasks, concurrency);
        return out;
    }

    const fill = nodata(layer);
    if (fill !== null && fill !== 0) out.fill(fill);  // chunks never written hold the fill value
    const [chunkBands, chunkRows, chunkCols] = layer.chunks.length === 3 ? layer.chunks : [1, ...layer.chunks];
    const tasks = layer.index.map(entry => async () => {
        const [band0, row0, col0, offset, length, mask] = entry.length === 6 ? entry : [0, ...entry];
        let bytes = await fetchRange(url, offset, length, signal);
        for (let i = layer.filters.length - 1; i >= 0; i--) {  // undo in reverse order of writing
            if (mask & (1 << i)) continue;  // filter skipped for this chunk
            const filter = layer.filters[i];
            if (filter === 'deflate') bytes = await inflate(bytes);
            else if (filter === 'shuffle') bytes = unshuffle(bytes, size);
            else if (filter === 'fletcher32') bytes = bytes.slice(0, bytes.length - 4);
        }
        const chunk = new Typed(bytes.buffer, bytes.byteOffset, chunkBands * chunkRows * chunkCols);
        const cols = Math.min(chunkCols, width - col0);
        for (let b = 0; b < chunkBands && band0 + b < bands; b++) {
            for (let r = 0; r < chunkRows && row0 + r < height; r++) {
                const src = (b * chunkRows + r) * chunkCols;
                out.set(chunk.subarray(src, src + cols), ((band0 + b) * height + row0 + r) * width + col0);
            }
        }
        progress(length);
    });
    await pool(tasks, concurrency);
    return out;
}

// --- GeoTIFF writing -----------------------------------------------------------------------------

function overviews(values, width, height, Typed, spp) {
    // Nearest-neighbour halvings until the image fits in one tile (safe for categorical layers too).
    // values are pixel-interleaved: spp samples (bands) per pixel.
    const levels = [{ values, width, height }];
    while (levels.at(-1).width > TILE || levels.at(-1).height > TILE) {
        const prev = levels.at(-1);
        const w = Math.ceil(prev.width / 2);
        const h = Math.ceil(prev.height / 2);
        const v = new Typed(w * h * spp);
        for (let y = 0; y < h; y++) {
            const src = 2 * y * prev.width;
            for (let x = 0; x < w; x++) {
                for (let s = 0; s < spp; s++) v[(y * w + x) * spp + s] = prev.values[(src + 2 * x) * spp + s];
            }
        }
        levels.push({ values: v, width: w, height: h });
    }
    return levels;
}

function predictFloat(bytes, width, size) {
    // TIFF predictor 3 (libtiff fpDiff): per row, bytes regrouped most significant first, then differenced.
    const out = new Uint8Array(bytes.length);
    const rowBytes = width * size;
    for (let base = 0; base < bytes.length; base += rowBytes) {
        for (let i = 0; i < width; i++) {
            for (let k = 0; k < size; k++) out[base + k * width + i] = bytes[base + i * size + (size - 1 - k)];
        }
        for (let j = rowBytes - 1; j > 0; j--) out[base + j] = (out[base + j] - out[base + j - 1]) & 0xff;
    }
    return out;
}

function predictInt(tile, width) {
    // TIFF predictor 2: horizontal differencing of samples, per row.
    const out = tile.slice();
    for (let base = 0; base < tile.length; base += width) {
        for (let i = width - 1; i > 0; i--) out[base + i] = tile[base + i] - tile[base + i - 1];
    }
    return new Uint8Array(out.buffer);
}

async function encodeTiles(level, Typed, format, pad, spp, onTile, signal) {
    const across = Math.ceil(level.width / TILE);
    const down = Math.ceil(level.height / TILE);
    const tiles = [];
    for (let ty = 0; ty < down; ty++) {
        for (let tx = 0; tx < across; tx++) {
            if (signal && signal.aborted) throw new DOMException('cancelled', 'AbortError');
            const tile = new Typed(TILE * TILE * spp);
            if (pad !== 0) tile.fill(pad);
            const cols = Math.min(TILE, level.width - tx * TILE);
            for (let r = 0; r < TILE && ty * TILE + r < level.height; r++) {
                const src = ((ty * TILE + r) * level.width + tx * TILE) * spp;
                tile.set(level.values.subarray(src, src + cols * spp), r * TILE * spp);
            }
            // Several bands (small 3-D layers) are written without a predictor.
            const raw = spp > 1 ? new Uint8Array(tile.buffer)
                : format === 3 ? predictFloat(new Uint8Array(tile.buffer), TILE, Typed.BYTES_PER_ELEMENT)
                    : predictInt(tile, TILE);
            tiles.push(await deflate(raw));
            onTile();
        }
    }
    return tiles;
}

function xmlEscape(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const TYPE_SIZE = { 2: 1, 3: 2, 4: 4, 12: 8 };  // ASCII, SHORT, LONG, DOUBLE

function ascii(s) {
    return Array.from(new TextEncoder().encode(s + '\0'));
}

function ifdBytes(entries, offset, nextOffset) {
    // entries: [tag, type, values]; returns the IFD followed by its out-of-line values.
    entries.sort((a, b) => a[0] - b[0]);
    const head = 2 + 12 * entries.length + 4;
    let extra = 0;
    for (const [, type, values] of entries) {
        const n = values.length * TYPE_SIZE[type];
        if (n > 4) extra += n + (n % 2);
    }
    const buf = new ArrayBuffer(head + extra);
    const view = new DataView(buf);
    view.setUint16(0, entries.length, true);
    let data = offset + head;
    const put = (pos, type, values) => {
        values.forEach((v, i) => {
            const p = pos + i * TYPE_SIZE[type];
            if (type === 2) view.setUint8(p, v);
            else if (type === 3) view.setUint16(p, v, true);
            else if (type === 4) view.setUint32(p, v, true);
            else view.setFloat64(p, v, true);
        });
    };
    entries.forEach(([tag, type, values], i) => {
        const pos = 2 + 12 * i;
        view.setUint16(pos, tag, true);
        view.setUint16(pos + 2, type, true);
        view.setUint32(pos + 4, values.length, true);
        const n = values.length * TYPE_SIZE[type];
        if (n <= 4) {
            put(pos + 8, type, values);
        } else {
            view.setUint32(pos + 8, data, true);
            put(data - offset, type, values);
            data += n + (n % 2);
        }
    });
    view.setUint32(2 + 12 * entries.length, nextOffset, true);
    return new Uint8Array(buf);
}

function geoKeys(epsg) {
    const geographic = epsg >= 4000 && epsg < 5000;
    return [1, 1, 0, 3, 1024, 0, 1, geographic ? 2 : 1, 1025, 0, 1, 1, geographic ? 2048 : 3072, 0, 1, epsg];
}

// Writes values (row-major, layer.shape) as a COG. Returns the file as an array of Uint8Array parts.
export async function encodeCOG(values, layer, { source = '', onProgress = () => {}, signal } = {}) {
    const [Typed, format, bits] = DTYPES[layer.dtype];
    const [spp, height, width] = dims(layer);
    let [x0, dx, , y0, , dy] = layer.geotransform;
    if (spp > 1) {  // band after band -> pixel-interleaved
        const interleaved = new Typed(values.length);
        const n = height * width;
        for (let b = 0; b < spp; b++) {
            for (let i = 0; i < n; i++) interleaved[i * spp + b] = values[b * n + i];
        }
        values = interleaved;
    }
    if (dy > 0) {  // south-up grid: flip to north-up
        const row = width * spp;
        const flipped = new Typed(values.length);
        for (let r = 0; r < height; r++) flipped.set(values.subarray(r * row, (r + 1) * row), (height - 1 - r) * row);
        values = flipped;
        y0 += dy * height;
        dy = -dy;
    }
    const fill = nodata(layer);
    const pad = fill === null ? 0 : fill;
    const levels = overviews(values, width, height, Typed, spp);
    const total = levels.reduce((n, l) => n + Math.ceil(l.width / TILE) * Math.ceil(l.height / TILE), 0);
    let done = 0;
    for (const level of levels) {
        level.tiles = await encodeTiles(level, Typed, format, pad, spp, () => onProgress(++done / total), signal);
    }

    // Per band: its description and units; for a 3-D layer also its height, under the name GDAL's
    // netCDF driver gives it (NETCDF_DIM_<dimension>).
    const bandItems = Array.from({ length: spp }, (_, b) => {
        const level = layer.bands ? layer.bands.values[b] : null;
        const description = level === null ? layer.name
            : `${layer.name} at ${layer.bands.name.replace(/s?Meta$/, '')} ${level}${layer.bands.units ? ' ' + layer.bands.units : ''}`;
        return `<Item name="DESCRIPTION" sample="${b}" role="description">${xmlEscape(description)}</Item>`
            + (layer.units ? `<Item name="UNITTYPE" sample="${b}" role="unittype">${xmlEscape(layer.units)}</Item>` : '')
            + (level === null ? '' : `<Item name="NETCDF_DIM_${xmlEscape(layer.bands.name)}" sample="${b}">${level}</Item>`);
    }).join('');
    const metadata = '<GDALMetadata>'
        + bandItems
        + `<Item name="SOURCE_PRODUCT">${xmlEscape(source)}</Item>`
        + `<Item name="SOURCE_LAYER">${xmlEscape(layer.path)}</Item>`
        + (layer.description ? `<Item name="LAYER_DESCRIPTION">${xmlEscape(layer.description)}</Item>` : '')
        + '</GDALMetadata>';
    const nodataTag = fill === null ? null : (Number.isNaN(fill) ? 'nan' : String(fill));
    const entriesFor = (level, i, offsets) => {
        const e = [
            [254, 4, [i === 0 ? 0 : 1]],
            [256, 4, [level.width]],
            [257, 4, [level.height]],
            [258, 3, Array(spp).fill(bits)],
            [259, 3, [8]],  // Adobe deflate
            [262, 3, [1]],
            [277, 3, [spp]],
            [284, 3, [1]],  // pixel-interleaved
            [317, 3, [spp > 1 ? 1 : format === 3 ? 3 : 2]],
            [322, 3, [TILE]],
            [323, 3, [TILE]],
            [324, 4, offsets],
            [325, 4, level.tiles.map(t => t.length)],
            [339, 3, Array(spp).fill(format)],
        ];
        if (spp > 1) e.push([338, 3, Array(spp - 1).fill(0)]);  // ExtraSamples: unspecified
        if (nodataTag !== null) e.push([42113, 2, ascii(nodataTag)]);
        if (i === 0) {
            e.push([33550, 12, [dx, -dy, 0]]);
            e.push([33922, 12, [0, 0, 0, x0, y0, 0]]);
            e.push([34735, 3, geoKeys(layer.epsg)]);
            e.push([42112, 2, ascii(metadata)]);
        }
        return e;
    };

    // Layout: header, all IFDs (main first), then tile data from the smallest overview to the main image.
    const zeros = level => level.tiles.map(() => 0);
    const ifdSizes = levels.map((level, i) => ifdBytes(entriesFor(level, i, zeros(level)), 0, 0).length);
    const ifdOffsets = [];
    let pos = 8;
    for (const n of ifdSizes) {
        ifdOffsets.push(pos);
        pos += n;
    }
    const tileOffsets = levels.map(() => []);
    for (let i = levels.length - 1; i >= 0; i--) {
        for (const t of levels[i].tiles) {
            tileOffsets[i].push(pos);
            pos += t.length;
        }
    }
    if (pos >= 2 ** 32) throw new Error('layer too large for a classic TIFF');
    const header = new Uint8Array(8);
    new DataView(header.buffer).setUint16(0, 0x4949, true);
    new DataView(header.buffer).setUint16(2, 42, true);
    new DataView(header.buffer).setUint32(4, 8, true);
    const parts = [header];
    levels.forEach((level, i) => {
        parts.push(ifdBytes(entriesFor(level, i, tileOffsets[i]), ifdOffsets[i], ifdOffsets[i + 1] || 0));
    });
    for (let i = levels.length - 1; i >= 0; i--) parts.push(...levels[i].tiles);
    return parts;
}

// --- helpers for the viewer ----------------------------------------------------------------------

export function productName(url) {
    return url.split('/').pop().replace(/\.nc$/, '');
}

export function fileName(url, layer) {
    return `${productName(url)}_${layer.name}.tif`;
}

export function gdalCommand(url, layer) {
    return `gdal_translate -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES 'NETCDF:"/vsicurl/${url}":${layer.path}' ${fileName(url, layer)}`;
}

// The product's bounding box (productBoundingBox in the netCDF; "footprint" in its layers.json).
export function footprintFileName(url) {
    return `${productName(url)}_productBoundingBox.geojson`;
}

export function footprintGeoJSON(url, footprint, properties = {}) {
    return JSON.stringify({
        type: 'FeatureCollection',
        features: [{ type: 'Feature', properties: { product: productName(url), source_url: url, ...properties }, geometry: footprint }],
    }, null, 1) + '\n';
}

export function footprintCommand(url) {
    return `ogr2ogr -f GeoJSON ${footprintFileName(url)} '/vsicurl/${url}' productBoundingBox`;
}

// Fetch, convert and return the COG parts; onProgress(phase, fraction) with phase 'fetch' or 'encode'.
export async function layerToCOG(url, layer, { onProgress = () => {}, signal, concurrency } = {}) {
    const values = await readLayer(url, layer, { concurrency, signal, onProgress: f => onProgress('fetch', f) });
    return encodeCOG(values, layer, { source: url, signal, onProgress: f => onProgress('encode', f) });
}
