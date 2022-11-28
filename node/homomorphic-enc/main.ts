import IPCIDR from 'ip-cidr';
import { WebSocketServer, WebSocket } from 'ws';
import SEAL from 'node-seal'
import { SEALLibrary } from 'node-seal/implementation/seal';
import { Context } from 'node-seal/implementation/context';
import * as fs from 'fs';
import { secretBase64Key } from './secret';
import { PlainText } from 'node-seal/implementation/plain-text';
import { BatchEncoder } from 'node-seal/implementation/batch-encoder';
import { Evaluator } from 'node-seal/implementation/evaluator';
// import { secretBase64Key } from './secret';
// import { secretKeyBase64Encoded } from './secret';

// The target value. Represents a "completed" federated ML model
let goal = 100;
let current = 0;

let name = '';

export default async function main() {
    name = process.env.NAME || 'unknown';


    // ---- Homomorphic Encryption Setup
    // See: https://github.com/morfix-io/node-seal/blob/main/USAGE.md
    const seal = await SEAL()
    const enc = setupHomomorphicEncryption(seal);
    const { context, } = enc;
    // -----------------------------
    
    // const cidr = new IPCIDR("10.151.0.0/28"); 
    // console.log(cidr.toArray());

    const listener = setupListener();
    
    // Wait 3 seconds
    await new Promise(resolve => setTimeout(resolve, 3000));

    const peer = new WebSocket(`ws://${process.env.PEER}:8080`, { 'timeout': 5000 });
    peer.onopen = () => {
        peer.send('Hello from ' + name);

        // Randomly send a message to the peer
        setInterval(async () => {
            await doWork(peer, context, seal, enc);
        }, 5000);
    }

    console.log(`I am ${name}!`);
}

async function doWork(peer: WebSocket, context: Context, seal: SEALLibrary, enc: { context: Context; encoder: BatchEncoder; evaluator: Evaluator; }) {
    //// Send a number between 0 and 20, 
    //// representing a piece of the Federated machine learning model solved by a node.
    // peer.send();
    
    const encodedPlaintext = seal.PlainText({ capacity: 10, coeffCount: 10});
    enc.encoder.encode(Uint32Array.from([1]), encodedPlaintext);

    const encodedPlaintextString = encodedPlaintext.save();
    console.log(encodedPlaintextString)

    const decodedPlaintext = enc.encoder.decode(encodedPlaintext).slice(0, 10);
    const decodedPlaintextString = decodedPlaintext.toString();
    console.log('decoded: ', decodedPlaintext)
    console.log('decoded_string: ', decodedPlaintextString)

}

function setupHomomorphicEncryption(seal: SEALLibrary) {   
    console.log('Setting up homomorphic encryption...');
    const schemeType = seal.SchemeType.bfv
    const securityLevel = seal.SecurityLevel.tc128
    const polyModulusDegree = 4096
    const bitSizes = [36, 36, 37]
    const bitSize = 20
    
    const encParms = seal.EncryptionParameters(schemeType)
    
    // Set the PolyModulusDegree
    encParms.setPolyModulusDegree(polyModulusDegree)
    
    // Create a suitable set of CoeffModulus primes
    encParms.setCoeffModulus(
      seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes))
    )
    
    // Set the PlainModulus to a prime of bitSize 20.
    encParms.setPlainModulus(seal.PlainModulus.Batching(polyModulusDegree, bitSize))


    // Create a new Context
    const context = seal.Context(
        encParms, // Encryption Parameters
        true, // ExpandModChain
        securityLevel // Enforce a security level
    )
        
    if (!context.parametersSet()) {
        throw new Error(
        'Could not set the parameters in the given context. Please try different encryption parameters.'
        )
    }

    //  --- Init nodes with the same secret.

    // Uploading a SecretKey: first, create an Empty SecretKey to load
    const UploadedSecretKey = seal.SecretKey()
    // Load from the base64 encoded string
    UploadedSecretKey.load(context, secretBase64Key)
    // Create a new KeyGenerator (use uploaded secretKey)
    const keyGenerator = seal.KeyGenerator(context, UploadedSecretKey)

    const encoder = seal.BatchEncoder(context)
    const evaluator = seal.Evaluator(context)

    // generateKeys(seal, context);

    return { context, encoder, evaluator };
}

// function string2array(s: string) {
//     return Uint32Array.from(s, (c) => c.codePointAt(0));
// }

function generateKeys (seal: SEALLibrary, context: Context) {
    const keyGenerator = seal.KeyGenerator(context)
    const secretKey = keyGenerator.secretKey()
    const publicKey = keyGenerator.createPublicKey()

    const secretBase64Key = secretKey.save()

    fs.writeFileSync('secret', secretBase64Key);

    console.log('Secret Key: \n', secretBase64Key);
    console.log('\n\n');

    const publicBase64Key = publicKey.save()

    console.log('Public Key: \n', publicBase64Key);
}


function setupListener() {
    const wss = new WebSocketServer({ port: 8080 });

    wss.on('connection', function connection(ws) {
        ws.on('message', function message(data) {
            const puzzlePiece = Number.parseInt(data.toString())
            if (!isNaN(puzzlePiece)) {
                console.log(`recv: ${puzzlePiece}`);
                current += puzzlePiece;
            }
            if (current >= goal) {
                console.log(`I am '${name}' and I have solved the puzzle!`);
            }
        });
    });

    return wss;
}



main();