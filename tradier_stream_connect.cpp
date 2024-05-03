#include <iostream>
#include <string>
#include <curl/curl.h>
#include <thread>
#include <unistd.h>
#include <cstdlib>

#include <nlohmann/json.hpp>

using json = nlohmann::json;


//
// Callback function - handles data returned from curl
//

size_t callback(const char* in, size_t size, size_t num, std::string* out) {
	const size_t totalBytes(size*num);
	out->append(in, totalBytes);
	return totalBytes;
}


//
// Initialize + Processs HTTP Request
//

void tradier_http_stream() {
	CURL* curl = curl_easy_init();
	if (curl) {
		std::string response_string;
		std::string header_string;
		curl_easy_setopt(curl, CURLOPT_URL, "https://api.tradier.com/v1/markets/events/session");

		// Authorization, headers
		// std::string auth_header = "Authorization: Bearer " + std::string(getenv("TRADIER_TOKEN_LIVE"));
		std::string auth_header = "Authorization: Bearer ABC123"; // replace ABC123 with actual authorization token
		// std::string auth_header = "Authorization: Bearer " + std::string(getenv("tradier_token_live"));
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, auth_header.c_str());
		headers = curl_slist_append(headers, "Accept: application/json");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		// Setup the POST request
		curl_easy_setopt(curl, CURLOPT_POST, 1L);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

		// Make the CURL request
		CURLcode res = curl_easy_perform(curl);
		if (res == CURLE_OK) {
			long http_code = 0;
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
			if (http_code == 200) {
				// Parse JSON
				json result = json::parse(response_string);
				auto stream_url = result["stream"]["url"].get<std::string>();
				auto session_id = result["stream"]["sessionid"].get<std::string>();

				std::cout << "Connect to stream : " << stream_url << " with session ID " << session_id << "\n\n";
				sleep(240);
			} else {
				std::cout << "Failed to connect.\nHTTP Status Code: " << http_code << "\n";
			}
		} else {
			std::cout << "CURL Error: " << curl_easy_strerror(res) << "\n";
		}

		curl_easy_cleanup(curl);
		curl_slist_free_all(headers);
	}
}


int main() {
	const char* env_var = std::getenv("tradier_token_live");
	std::cout << "env var: " << env_var << std::endl;
	std::cout << "hello, world!" << std::endl;
	std::thread t(tradier_http_stream);
	t.join();
	return 0;
}




//
// OUTPUT
//

// thammons@toms-MacBook-Air t_bot % g++ -std=c++17 -o tradier_stream_connect tradier_stream_connect.cpp -I/opt/homebrew/Cellar/nlohmann-json/3.11.3/include -lcurl -lpthread
// thammons@toms-MacBook-Air t_bot % ./tradier_stream_connect 
// hello, world!
// Connect to stream : https://stream.tradier.com/v1/markets/events with session ID a7344a1f-cb26-4528-9c76-03224c3ed368
