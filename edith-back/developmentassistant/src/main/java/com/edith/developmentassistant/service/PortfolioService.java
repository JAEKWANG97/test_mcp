package com.edith.developmentassistant.service;

import com.edith.developmentassistant.client.dto.UserDto;
import com.edith.developmentassistant.client.user.UserServiceClient;
import com.edith.developmentassistant.domain.Portfolio;
import com.edith.developmentassistant.domain.UserProject;
import com.edith.developmentassistant.repository.MRSummaryRepository;
import com.edith.developmentassistant.repository.PortfolioRepository;
import com.edith.developmentassistant.repository.UserProjectRepository;
import com.edith.developmentassistant.service.dto.MergeRequest;
import com.edith.developmentassistant.service.dto.MergeRequestDateRange;
import com.edith.developmentassistant.service.dto.PortfolioDto;
import com.edith.developmentassistant.service.dto.Summary;
import com.edith.developmentassistant.service.dto.request.CreatePortfolioServiceRequest;
import com.edith.developmentassistant.service.dto.response.FindAllPortfolioResponse;
import com.edith.developmentassistant.service.dto.response.FlaskPortfolioResponse;
import com.edith.developmentassistant.service.dto.response.GitLabMergeRequestResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@Slf4j
@RequiredArgsConstructor
@Transactional
public class PortfolioService {

    private final PortfolioRepository portfolioRepository;
    private final UserProjectRepository userProjectRepository;
    private final MRSummaryRepository mrSummaryRepository;
    private final UserServiceClient userServiceClient;
    private final WebClient gitLabWebClient;
    private final RestTemplate portfolioRestTemplate;
    private final ProjectService projectService;

    @Value("${api.flask.portfolio}")
    String FLASK_PORTFOLIO_URL;

    // Portfolio 생성 로직
    public PortfolioDto createPortfolio(String accessToken, String projectId, String branch) {

        try {
            // 1. User, userProject 찾기
            UserDto user = userServiceClient.getUserByToken(accessToken);
//            UserDto user = createUserDto();
            UserProject userProject = projectService.findUserProjectByUserIdAndProjectId(user.getUserId(), Long.parseLong(projectId));
//            UserProject userProject = createUserProject();

            // 2. project summery 찾기 -> id 로 찾기
            List<Summary> summaries = mrSummaryRepository.findByProjectId(Long.parseLong(projectId)).stream()
                    .map(Summary::from)
                    .toList();

            // 3. GitLab 에서 해당 Branch 의 MR 리스트 받아 파싱하기 (WebClient)
            MergeRequestDateRange mergeRequestdateRange = getMergedMRs(projectId, branch, "NHMeAABxUvZVyLq6u5Qx");
            log.info("Merged MRs from {}", mergeRequestdateRange);
            // 4. Flask 에 포폴 생성 요청하기
            CreatePortfolioServiceRequest request = new CreatePortfolioServiceRequest(
                    "doublehyun98@gmail.com",
                    summaries,
                    mergeRequestdateRange.getMergeRequests()
            );
//            log.info("Create portfolio request: {}", request);
            // 5. 포트폴리오 받아 반환하기
            ResponseEntity<FlaskPortfolioResponse> response = portfolioRestTemplate.postForEntity(
                    FLASK_PORTFOLIO_URL,
                    request,
                    FlaskPortfolioResponse.class // 응답 타입
            );

            return PortfolioDto.builder()
                    .portfolio(response.getBody().getPortfolio())
                    .projectId(Long.parseLong(projectId))
                    .name(userProject.getTitle())
                    .content(userProject.getDescription())
                    .endDate(mergeRequestdateRange.getEndDate().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")))
                    .startDate(mergeRequestdateRange.getStartDate().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")))
                    .build();

        } catch (Exception e) {
            log.error(e.getMessage());
            throw new RuntimeException(e);
        }
    }

    public MergeRequestDateRange getMergedMRs(String projectId, String branch, String accessToken) {
        return gitLabWebClient
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/projects/{id}/merge_requests")
                        .queryParam("target_branch", branch)
                        .queryParam("state", "merged")
                        .queryParam("per_page", 100000000)
                        .build(projectId))
                .header("PRIVATE-TOKEN", accessToken)
                .retrieve()
                .bodyToFlux(GitLabMergeRequestResponse.class)
                .collectList()
                .flatMap(mergeRequests -> {
                    LocalDateTime firstPreparedAt = mergeRequests.stream()
                            .map(GitLabMergeRequestResponse::getPreparedAt)
                            .filter(Objects::nonNull)
                            .min(LocalDateTime::compareTo)
                            .orElse(null);

                    LocalDateTime lastMergedAt = mergeRequests.stream()
                            .map(GitLabMergeRequestResponse::getMergedAt)
                            .filter(Objects::nonNull)
                            .max(LocalDateTime::compareTo)
                            .orElse(null);

                    return Flux.fromIterable(mergeRequests)
                            .parallel()
                            .runOn(Schedulers.boundedElastic())
                            .flatMap(mr -> getMRDiff(projectId, mr.getIid(), accessToken)
                                    .map(diff -> new MergeRequest(
                                            String.valueOf(mr.getIid()),
                                            mr.getAuthor() != null ? mr.getAuthor().getUsername() : "",
                                            Optional.ofNullable(mr.getChanges())
                                                    .orElse(Collections.emptyList())
                                                    .stream()
                                                    .map(GitLabMergeRequestResponse.Change::getNewPath)
                                                    .collect(Collectors.joining(",")),
                                            diff
                                    )))
                            .sequential()
                            .collectList()
                            .map(mrs -> new MergeRequestDateRange(firstPreparedAt, lastMergedAt, mrs));
                })
                .onErrorResume(e -> {
                    log.error("Error fetching merge requests: ", e);
                    return Mono.just(new MergeRequestDateRange(null, null, Collections.emptyList()));
                })
                .block(Duration.ofSeconds(30));
    }

    public PortfolioDto savePortfolio(String accessToken, PortfolioDto portfolio) {

        UserDto user = userServiceClient.getUserByToken(accessToken);
        UserProject userProject = projectService.findUserProjectByUserIdAndProjectId(user.getUserId(), portfolio.getProjectId());

        Optional<Portfolio> existingPortfolio = portfolioRepository.findByUserProject(userProject);
        PortfolioDto savedPortfolio;
        if (existingPortfolio.isPresent()) {
            // 기존 portfolio 업데이트
            Portfolio portfolioToUpdate = existingPortfolio.get();
            portfolioToUpdate.updateContent(portfolio.getPortfolio());  // 내용 업데이트 메소드 필요
            portfolioToUpdate.updateDates(  // 날짜 업데이트 메소드 필요
                    LocalDate.parse(portfolio.getStartDate()).atStartOfDay(),
                    LocalDate.parse(portfolio.getEndDate()).atStartOfDay()
            );
            savedPortfolio = new PortfolioDto(userProject, portfolioRepository.save(portfolioToUpdate));
        } else {
            // 새로운 portfolio 생성
            Portfolio savePortfolio = Portfolio.builder()
                    .content(portfolio.getPortfolio())
                    .userProject(userProject)
                    .startDate(LocalDate.parse(portfolio.getStartDate()).atStartOfDay())
                    .endDate(LocalDate.parse(portfolio.getEndDate()).atStartOfDay())
                    .build();
            savedPortfolio = new PortfolioDto(userProject, portfolioRepository.save(savePortfolio));
        }

        return savedPortfolio;
    }

    public List<FindAllPortfolioResponse> findAllPortfolioResponseList(String accessToken) {
        UserDto user = userServiceClient.getUserByToken(accessToken);
        return portfolioRepository.findAllDtoByUserId(user.getUserId());
    }

    public PortfolioDto getPortfolio(String accessToken, String portfolioId) {

        UserDto user = userServiceClient.getUserByToken(accessToken);
        UserProject userProject = userProjectRepository.findByUserIdAndProjectId(user.getUserId(), Long.parseLong(portfolioId))
                .orElse(null);

        Portfolio portfolio = portfolioRepository.findByUserProject(userProject)
                .orElse(null);

        if (portfolio == null || userProject == null) {
            return null;
        }
        return new PortfolioDto(userProject, portfolio);
    }

    private Mono<String> getMRDiff(String projectId, Long mrIid, String accessToken) {
        return gitLabWebClient
                .get()
                .uri("/projects/{projectId}/merge_requests/{mrIid}/changes", projectId, mrIid)
                .header("PRIVATE-TOKEN", accessToken)
                .retrieve()
                .bodyToMono(GitLabMergeRequestResponse.class)
                .map(mr -> Optional.ofNullable(mr.getChanges())
                        .orElse(Collections.emptyList())
                        .stream()
                        .map(change -> Optional.ofNullable(change.getDiff()).orElse(""))
                        .collect(Collectors.joining("\n")))
                .onErrorResume(e -> {
                    log.error("Error fetching MR diff: ", e);
                    return Mono.just("");
                })
                .timeout(Duration.ofSeconds(10));
    }

//    private UserDto createUserDto() {
//        return UserDto.builder()
//                .userId(1L)
//                .email("doublehyun98")
//                .password("1234")
//                .vcsBaseUrl("https://lab.ssafy.com/")
//                .vcsAccessToken("ZH3_Ft1HJmHqwXYmgYHs")
//                .build();
//    }
//
//    private UserProject createUserProject() {
//
//        Project project = Project.builder()
//                .projectId(824085L)
//                .build();
//
//
//        return UserProject.builder()
//                .userId(1L)
//                .name("E.D.I.T.H.")
//                .description("asdfasdf")
//                .project(project)
//                .build();
//    }




}


