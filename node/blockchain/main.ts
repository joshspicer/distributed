import IPCIDR from 'ip-cidr';
import { WebSocketServer, WebSocket } from 'ws';

// The target value. Represents a "completed" federated ML model
let goal = 100;
let current = 0;

let name = '';

export default async function main() {
    name = process.env.NAME || 'unknown';
    
    // const cidr = new IPCIDR("10.151.0.0/28"); 
    // console.log(cidr.toArray());

    const listener = setupListener();
    
    // Wait 3 seconds
    await new Promise(resolve => setTimeout(resolve, 3000));

    const peer = new WebSocket(`ws://${process.env.PEER}:8080`, { 'timeout': 5000 });
    peer.onopen = () => {
        peer.send('Hello from ' + name);

        // Randomly send a message to the peer
        setInterval(() => {
            // Send a number between 0 and 20, 
            // representing a piece of the Federated machine learning model solved by a node.
            peer.send(Math.trunc(Math.random() * 20));
        }, 1000);
    }

    console.log(`I am ${name}!`);
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