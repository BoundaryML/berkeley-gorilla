from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

LLM_MODEL = os.environ['BAML_CLIENT'].split(':')[1]
RESULT_DIR = f'result/{LLM_MODEL}'

class TraceRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/log/v2':
            content_length = int(self.headers['Content-Length'])  # Get the size of data
            post_data = self.rfile.read(content_length)  # Read the data
            data = json.loads(post_data.decode('utf-8'))  # Decode it to string

            test_category = data['context']['tags']['test_category']
            print(f"Received event log for {test_category}: {data['context']['event_chain'][0]['function_name']}")
            #print(json.dumps(data, indent=2))
            with open(os.path.join(RESULT_DIR, f"{test_category}_{os.environ['REPORT_DATE']}.jsonl"), "a") as f:
                f.write(json.dumps(data) + '\n')

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'message': 'Log received'})
            self.wfile.write(response.encode('utf-8'))  # Send response back to client
        else:
            self.send_error(404, "File not found")

if __name__ == '__main__':
    address = ('', int(os.environ['BOUNDARY_BASE_URL'].rsplit(':', maxsplit=1)[1]))
    httpd = HTTPServer(address, TraceRequestHandler)
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f'Starting BAML event logger on {address}')
    httpd.serve_forever()
