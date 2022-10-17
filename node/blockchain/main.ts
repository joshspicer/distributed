import IPCIDR from 'ip-cidr';
import { WebSocketServer, WebSocket } from 'ws';


export default async function main() {
    const name = process.env.NAME;
    
    // const cidr = new IPCIDR("10.151.0.0/28"); 
    // console.log(cidr.toArray());

    const listener = setupListener();

    const peer = new WebSocket(`ws://${process.env.PEER}:8080`);
    peer.onopen = () => {
        peer.send('Hello from ' + name);

        // Randomly send a message to the peer
        setInterval(() => {
            peer.send('Random ping from ' + name);
        }, Math.random() * 3000);
    }

    console.log(`I am ${name}!`);
}


function setupListener() {
    const wss = new WebSocketServer({ port: 8080 });

    wss.on('connection', function connection(ws) {
        ws.on('message', function message(data) {
            console.log('received: %s', data);
        });
    });

    return wss;
}



main();