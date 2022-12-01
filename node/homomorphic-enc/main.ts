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
import { Encryptor } from 'node-seal/implementation/encryptor';
import { Decryptor } from 'node-seal/implementation/decryptor';
import { CipherText } from 'node-seal/implementation/cipher-text';
// import { secretBase64Key } from './secret';
// import { secretKeyBase64Encoded } from './secret';

// The target value. Represents a "completed" federated ML model
// The data is always stored encrypted for each node.
let goal: number = 100;
let current: CipherText | undefined | void = undefined; 

let enc:  {
    context: Context;
    encoder: BatchEncoder;
    evaluator: Evaluator;
    encryptor: Encryptor;
    decryptor: Decryptor;
} | undefined = undefined;

let seal: SEALLibrary | undefined = undefined;
let name = '';

export default async function main() {
    name = process.env.NAME || 'unknown';
    console.log(`I am ${name}!`);

    // ---- Homomorphic Encryption Setup
    // See: https://github.com/morfix-io/node-seal/blob/main/USAGE.md
    seal = await SEAL()
    enc = setupHomomorphicEncryption(seal);
    if (!enc || !seal) {
        throw new Error('Failed to setup encryption!');
    }
    const { context, } = enc;
    // -----------------------------
    
    // const cidr = new IPCIDR("10.151.0.0/28"); 
    // console.log(cidr.toArray());

    const _ = setupListener();
    
    // Wait 3 seconds for the other nodes to wake up.
    await new Promise(resolve => setTimeout(resolve, 3000));

    const peer = new WebSocket(`ws://${process.env.PEER}:8080`, { 'timeout': 5000 });

    peer.onopen = () => {
        // peer.send('Hello from ' + name);

        // Randomly send a message to the peer
        setInterval(async () => {
            await doWork(peer, context, seal, enc);
        }, (Math.random() + 5) * 1000);
    }

}

async function doWork(peer: WebSocket, context: Context, seal: SEALLibrary | undefined, enc: { context: Context; encoder: BatchEncoder; evaluator: Evaluator; encryptor: Encryptor; decryptor: Decryptor} | undefined) {
    //// Send a number between 0 and 20, 
    //// representing a piece of the Federated machine learning model solved by a node.

    if (!enc || !seal) {
        throw new Error('Homomorphic encryption not setup!');
    }

    const generatePayload = () => {
        const list: number[] = [];
        for (let i = 0; i < 10; i++) {
            list.push(Math.floor(Math.random() * 20));
        }
        console.log('Generated values to encrypt and send to peer: ', list.slice(0, 5));
        return list;
    }
    
    const encodedPlaintext = seal.PlainText();
    enc.encoder.encode(Uint32Array.from(generatePayload()), encodedPlaintext);
    const cipherText = enc.encryptor.encrypt(encodedPlaintext)

   if (!cipherText) {
         throw new Error('Failed to encrypt!');
   }

    const cipherTextString = cipherText.save();
    // SEND TO PEER
    peer.send(cipherTextString);
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
    const publicKey = keyGenerator.createPublicKey()


    const encoder = seal.BatchEncoder(context)
    const evaluator = seal.Evaluator(context)

    const encryptor = seal.Encryptor(context, publicKey)
    const decryptor = seal.Decryptor(context, UploadedSecretKey)

    // generateKeys(seal, context);

    return { context, encoder, evaluator, encryptor, decryptor };
}

// function string2array(s: string) {
//     return Uint32Array.from(s, (c) => c.codePointAt(0));
// }

// function generateKeys (seal: SEALLibrary, context: Context) {
//     const keyGenerator = seal.KeyGenerator(context)
//     const secretKey = keyGenerator.secretKey()
//     const publicKey = keyGenerator.createPublicKey()

//     const secretBase64Key = secretKey.save()

//     fs.writeFileSync('secret', secretBase64Key);

//     // console.log('Secret Key: \n', secretBase64Key);
//     // console.log('\n\n');
//     const publicBase64Key = publicKey.save()
//     // console.log('Public Key: \n', publicBase64Key);
// }

function setupListener() {
    const wss = new WebSocketServer({ port: 8080 });

    wss.on('connection', function connection(ws) {
        ws.on('message', function message(data) {

            if (!enc || !seal) {
                throw new Error('Homomorphic encryption not setup!');
            }

            console.log(`Recv'd encrypted payload: ...${data.toString().slice(data.toString().length - 20)}`);

            const recvCipherText = seal.CipherText();
            recvCipherText.load(enc.context, data.toString());

            // This is the first message received from the peer
            // Populate the 'current' CipherText object and simply exit to wait for the next message
            if (!current) {
                current = recvCipherText;
                return;
            }

            // Add received ciphertext to the 'current' encrypted value.
            current = enc.evaluator.add(recvCipherText, current)

            if (!current) {
                throw new Error('Failed to add current and incoming cipherTexts!');
            }

            // -- At this point computation on the cipher text has been completed without ever decrypting it.

            // Decrypt the result for demo purposes.
            const decrypted = enc.decryptor.decrypt(current)
            if (decrypted) {
                const decodedArray = enc.encoder.decode(decrypted)
                console.log(`Current state decrypted and decoded: ${decodedArray.slice(0, 5)}`);
            }
        });
    });

    return wss;
}

main();