import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'lil-gui';

// PDF.js.
import * as pdfjs from 'pdfjs-dist/build/pdf'; // Main PDF.js library
// It uses a background WebWorker for page rendering. We need to properly
// set up the URL for it so it could be loaded. There is additional fixup
// in vite.config.js so that the path is properly resolved in the builds
import pdfWorkerURL from 'pdfjs-dist/build/pdf.worker.min?url'
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  pdfWorkerURL,
  import.meta.url,
).toString();



const scene = new THREE.Scene();

// Configure Camera and Renderer
const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);
camera.position.set(0, 0, 5);

const renderer = new THREE.WebGLRenderer({
    canvas:    document.getElementById("three-canvas"),
    antialias: true,
});

renderer.setSize(window.innerWidth, window.innerHeight);

// Remember to adjust camera aspect ratio and renderer viewport on resizes
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.7;
controls.minZoom = 0.5;
controls.maxZoom = 3;

// Ambient light setup
const ambientLight = new THREE.AmbientLight(0xffffff);
scene.add(ambientLight);

// Two directional lights to keep models non-flat from all sides
const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight1.position.set(5, 5, 5);
scene.add(directionalLight1);
const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
directionalLight2.position.set(-5, -5, 5);
scene.add(directionalLight2);

const gridHelper = new THREE.GridHelper(10, 10);
gridHelper.position.y = 0;
scene.add(gridHelper);

// Floor setup
let buildingConfig = {};
const buildingFloors = [];

function getFloorHeight() {
    return buildingConfig.Elevators[0].Floor_to_floor_height_mm / 1000.0;
}

function getFloorBaseY(numFloor) {
    let y = 0;

    for (let i = 0; i < numFloor; i++) {
        if (buildingFloors[i] == null) {
            continue;
        }

        y += buildingFloors[i].height;
    }

    return y;
}

const guiObject = {
    currentFloor: null,
};

// GUI Setup
const gui = new GUI({
    title: 'Controls',
});

const guiFloorSelectorController = gui.add(guiObject, 'currentFloor', []);
guiFloorSelectorController.onChange((v) => {
    for (let i = 0; i <= v; i++) {
        if (buildingFloors[i] === null) {
            continue;
        }
        if (buildingFloors[i].plane !== null) {
            buildingFloors[i].plane.visible = true;
        }
    }
    for (let i = v + 1; i < buildingFloors.length; i++) {
        if (buildingFloors[i] === null) {
            continue;
        }

        if (buildingFloors[i].plane !== null) {
            buildingFloors[i].plane.visible = false;
        }
    }
});

// Render Loop
function animate() {
    // Request next frame right away
    requestAnimationFrame(animate);

    controls.update();

    renderer.render(scene, camera);
}
animate();

// Drag-and-Drop Handlers
renderer.domElement.addEventListener('dragover', (e) => e.preventDefault());

async function loadFloorPdfFile(floorNum, data) {
    const pdf = await pdfjs.getDocument({ data: data }).promise;

    const page = await pdf.getPage(1);
    const scale = 1.5;
    const viewport = page.getViewport({scale});

    const canvas = document.createElement('canvas');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const context = canvas.getContext('2d');
    await page.render({canvasContext: context, viewport}).promise;

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({map: texture});
    const geometry = new THREE.PlaneGeometry(viewport.width / 100, viewport.height / 100);

    const plane = new THREE.Mesh(geometry, material.clone());
    plane.rotation.x = -Math.PI / 2;
    plane.position.set(0, getFloorBaseY(floorNum), 0);
    scene.add(plane);

    if (floorNum > guiObject.currentFloor) {
        plane.visible = false;
    }
    buildingFloors[floorNum].plane = plane;
}

async function loadFloorFiles(config, files) {
    const numFloors = config['Number_of_floors'];
    let floors_found = [];
    let has_kellari = false;
    for (const [path, data] of files) {
        if (path == "kellari.pdf") {
            has_kellari = true;
        } else if (path.startsWith('floor_') && path.endsWith('.pdf')) {
            const strNum = path.match(/floor_(\d+)\.pdf/);
            const flrNum = parseInt(strNum[1], 10);
            floors_found.push(flrNum);
        }
    }
    floors_found = floors_found.sort((a, b) => a - b);

    while (buildingFloors.length > 0) {
        buildingFloors.pop();
    }

    if (has_kellari) {
        floors_found.unshift(0);
    } else {
        buildingFloors.push(null);
    }

    guiObject.currentFloor = 1;
    guiFloorSelectorController.options(floors_found);

    for (const floorNum of floors_found) {
        const path = floorNum === 0 ? "kellari.pdf" : `floor_${floorNum}.pdf`;

        buildingFloors.push({
            isGround: floorNum === 0,
            path:     path,
            plane:    null,
            height:   getFloorHeight(),

            absoluteHeight: 0.0,
        });

        buildingFloors[floorNum].absoluteHeight = getFloorBaseY(floorNum);
        console.log(floorNum, buildingFloors[floorNum].absoluteHeight);

        const data = await files.get(path).arrayBuffer();
        loadFloorPdfFile(floorNum, data);
    }
}

async function processRoot(files) {
    const fileMap = new Map();
    let siteConfig = '';
    for (const file of files) {
        if (!file.isFile) {
            continue;
        }

        const data = await new Promise((resolve) => {
            return file.file(resolve);
        });
        fileMap.set(file.name, data);
        if (file.name.endsWith('.json')) {
            siteConfig = file.name;
        }
    }

    const configText = await fileMap.get(siteConfig).text();
    const configData = JSON.parse(configText);
    
    const hasCoordinates = 'coordinates' in configData;
    if (hasCoordinates) {
        console.log('Coordinates: TODO');
        throw "something";
    }

    buildingConfig = configData;
    getFloorHeight();

    // has coordinates => load OSM
    // else load floor data
    const floors = await loadFloorFiles(configData, fileMap);
    console.log(configData);
}

renderer.domElement.addEventListener('drop', async (e) => {
    e.preventDefault();

    const rootDir = e.dataTransfer.items[0];
    const rootDirEntry = rootDir.webkitGetAsEntry();
    if (!rootDirEntry.isDirectory) {
        return;
    }

    rootDirEntry.createReader().readEntries((entries) => {
        const files = [];
        for (const entry of entries) {
            files.push(entry);
        }
        processRoot(files);
    });
    
    // const files = e.dataTransfer.files;
    // const gltfFile = Array.from(files).find(file => file.name.endsWith('.gltf'));
    // const binFile = Array.from(files).find(file => file.name.endsWith('.bin'));

    // if (gltfFile && binFile) {
    //     const gltf = await loadGLTFWithBin(gltfFile, binFile);
    //     if (gltf) {
    //         const model = gltf.scene;

    //         model.traverse((child) => {
    //             if (!child.isMesh) {
    //                 return;
    //             }

    //             child.material = new THREE.MeshStandardMaterial({
    //                 color: child.material.color,
    //                 roughness: 1,
    //                 metalness: 0,
    //             });
    //         });

    //         floors[activeFloor].add(model);

    //         model.position.set(Math.random() * 4 - 2, 0, Math.random() * 4 - 2); // Position in scene
    //     }
    // }
});

// Function to load .gltf with associated .bin from drag-and-drop files
async function loadGLTFWithBin(gltfFile, binFile) {
    return new Promise((resolve, reject) => {
        const gltfReader = new FileReader();
        const binReader = new FileReader();

        // Read the .bin file into a Blob URL
        binReader.onload = () => {
            const binUrl = URL.createObjectURL(new Blob([binReader.result], { type: 'application/octet-stream' }));

            // Read the .gltf file
            gltfReader.onload = () => {
                const gltfData = gltfReader.result;
                const gltfUrl = URL.createObjectURL(new Blob([gltfData], { type: 'application/json' }));

                const loader = new GLTFLoader();

                // Custom manager to resolve bin URL
                loader.manager = new THREE.LoadingManager();
                loader.manager.setURLModifier((url) => (url.endsWith('.bin') ? binUrl : url));

                loader.load(gltfUrl, (gltf) => {
                    resolve(gltf);
                    URL.revokeObjectURL(gltfUrl);
                    URL.revokeObjectURL(binUrl); // Cleanup URLs
                }, undefined, reject);
            };

            gltfReader.readAsArrayBuffer(gltfFile);
        };

        binReader.readAsArrayBuffer(binFile);
    });
}
