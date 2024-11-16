package com.edith.developmentassistant.service;

import com.edith.developmentassistant.client.dto.UserDto;
import com.edith.developmentassistant.client.dto.gitlab.ContributorDto;
import com.edith.developmentassistant.client.dto.gitlab.GitCommit;
import com.edith.developmentassistant.client.dto.gitlab.GitGraph;
import com.edith.developmentassistant.client.dto.gitlab.GitMerge;
import com.edith.developmentassistant.client.gitlab.GitLabServiceClient;
import com.edith.developmentassistant.client.user.UserServiceClient;
import com.edith.developmentassistant.controller.dto.response.project.ProjectDashboardDto;
import com.edith.developmentassistant.controller.dto.response.project.ProjectDto;
import com.edith.developmentassistant.controller.dto.response.project.ProjectResponse;
import com.edith.developmentassistant.controller.dto.response.project.ProjectStats;
import com.edith.developmentassistant.controller.dto.response.project.UsersProjectsStats;
import com.edith.developmentassistant.domain.Project;
import com.edith.developmentassistant.domain.UserProject;
import com.edith.developmentassistant.factory.ProjectFactory;
import com.edith.developmentassistant.repository.ProjectRepository;
import com.edith.developmentassistant.repository.UserProjectRepository;
import com.edith.developmentassistant.service.dto.DashboardDto;
import com.edith.developmentassistant.service.dto.request.RegisterProjectServiceRequest;
import jakarta.transaction.Transactional;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Slf4j
@Transactional
@Service
@RequiredArgsConstructor
public class ProjectService {

    private final ProjectRepository projectRepository;
    private final UserProjectRepository userProjectRepository;
    private final WebhookService webhookService;
    private final UserServiceClient userServiceClient;
    private final GitLabServiceClient gitLabServiceClient;
    private final RedisTemplate<String, Object> redisTemplate;

    public void registerProject(RegisterProjectServiceRequest request, String token) {

        UserDto user = userServiceClient.getUserByToken(token);
        String personalAccessToken = user.getVcsAccessToken();
        Long userId = user.getUserId();

        String projectAccessToken = gitLabServiceClient.generateProjectAccessToken(request.id(), personalAccessToken);

        Project project = getProjectByTokenAndProjectId(request, projectAccessToken);

        updateBranchesIfNeeded(project, request);

        log.info("request.name: {}", request.name());

        userProjectRepository.save(createUserProject(request, project, userId));
    }


    public List<ProjectResponse> getProjects(String token) {

        Long userId = getUserIdByToken(token);

        List<UserProject> userProjects = getUserProjectsBy(userId);

        List<Long> projectIds = userProjects.stream()
                .map(userProject -> userProject.getProject().getId())
                .toList();

        return getProjectsResponsesV2(token, projectIds);
    }

    public List<GitGraph> getGitGraphData(Long projectId, String accessToken) {

        UserDto userDto = userServiceClient.getUserByToken(accessToken);

        String projectAccessToken = gitLabServiceClient.generateProjectAccessToken(projectId,
                userDto.getVcsAccessToken());

        // 최근 MergeRequest목록 가져오기 (5개)
        List<GitMerge> merges = gitLabServiceClient.fetchGitLabMergeRequests(projectId, projectAccessToken);

        return merges.stream()
                .map(mergeRequest -> {
                    // MergeRequest 커밋 가져오기
                    GitCommit mergeCommit = gitLabServiceClient.fetchCommitDetails(projectId,
                            mergeRequest.getMerge_commit_sha(), projectAccessToken);

                    // 해당 MergeRequest내 전체 커밋을 가져오기
                    List<GitCommit> sourceBranchCommits = gitLabServiceClient.fetchCommitsInMergeRequest(
                            projectId,
                            mergeRequest.getIid(),
                            projectAccessToken
                    );

                    return GitGraph.of(
                            mergeRequest.getSource_branch(),
                            mergeRequest.getTarget_branch(),
                            mergeCommit,
                            sourceBranchCommits
                    );
                })
                .toList();
    }


    public UserProject findUserProjectByUserIdAndProjectId(Long userId, Long projectId) {
        return userProjectRepository.findByUserIdAndProjectId(userId, projectId)
                .orElse(null);
    }

    private Project createNewProject(RegisterProjectServiceRequest request, String personalAccessToken) {
        webhookService.registerWebhook(request, personalAccessToken);
        return projectRepository.save(ProjectFactory.createProject(request, personalAccessToken));
    }

    private void updateBranchesIfNeeded(Project project, RegisterProjectServiceRequest request) {
        project.updateBranches(request.branches().stream()
                .map(branchName -> ProjectFactory.createBranch(branchName, project))
                .toList());
    }


    public ProjectResponse getProjectByTokenAndProjectId(String token, Long projectId) {

        UserProject userProject = getUserProjectByProjectIdAndUserId(projectId, getUserIdByToken(token));

        Project project = userProject.getProject();

        List<ContributorDto> contributors = gitLabServiceClient.fetchContributors(project.getId(), project.getToken());

        return ProjectResponse.from(project, userProject.getTitle(), contributors, userProject.getDescription());
    }

    public ProjectResponse updateProject(ProjectDto projectDto, String token) {

        List<UserProject> userProjects = getUserProjectsBy(getUserIdByToken(token));

        String content = "";
        String projectName = "";

        Project projectToUpdate = null;

        UserProject userProject1 = userProjects.stream()
                .filter(userProject -> userProject.getProject().getId().equals(projectDto.id()))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Project not found for the user"));

        projectToUpdate = userProject1.getProject();
        projectName = userProject1.getTitle();
        content = userProject1.getDescription();

        if (projectToUpdate == null) {
            throw new IllegalArgumentException("Project not found for the user");
        }

        projectToUpdate.updateProject(projectDto);

        projectRepository.save(projectToUpdate);

        return ProjectResponse.from(projectToUpdate, projectName, null, content);
    }

    private List<UserProject> getUserProjectsBy(Long token) {
        return userProjectRepository.findByUserId(token).stream().toList();
    }


    private UserProject getUserProjectByProjectIdAndUserId(Long projectId, Long userId) {
        return userProjectRepository.findByUserIdAndProjectId(userId, projectId)
                .orElseThrow(() -> new IllegalArgumentException("Project not found for the user"));
    }

    private Long getUserIdByToken(String token) {
        UserDto userByToken = userServiceClient.getUserByToken(token);
        return userByToken.getUserId();
    }

    private String getUserEmailByToken(String token) {
        UserDto userByToken = userServiceClient.getUserByToken(token);
        return userByToken.getEmail();
    }

    private List<ProjectResponse> getProjectsResponses(List<UserProject> userProjects) {
        return userProjects.stream()
                .map(userProject -> {
                    Project project = userProject.getProject();
                    List<ContributorDto> contributors = gitLabServiceClient.fetchContributors(project.getId(),
                            project.getToken());
                    return ProjectResponse.from(project, userProject.getTitle(), contributors,
                            userProject.getDescription());
                })
                .toList();
    }

    private List<ProjectResponse> getProjectsResponsesV2(String token, List<Long> projectIds) {
        return projectIds.stream()
                .map(projectId -> getProjectByTokenAndProjectId(token, projectId))
                .toList();
    }

    private Project getProjectByTokenAndProjectId(RegisterProjectServiceRequest request, String projectAccessToken) {
        return projectRepository.findById(request.id())
                .orElseGet(() -> createNewProject(request, projectAccessToken));
    }

    private UserProject createUserProject(RegisterProjectServiceRequest request, Project project, Long userId) {
        log.info("Creating UserProject for userId: {}, id: {}, project-name: {}", userId, project.getId(),
                request.name());
        return UserProject.builder()
                .userId(userId)
                .description(request.contents())
                .title(request.name())
                .project(project)
                .build();
    }

    public UsersProjectsStats getUsersProjectsStats(String token) {
        log.info("developmentassisstant service에서 보내는 JWT Token: {}", token);
        Long userId = getUserIdByToken(token);
        String userEmail = getUserEmailByToken(token);

        List<UserProject> userProjects = getUserProjectsBy(userId);

        Integer totalProjectsCount = userProjects.size();

        Integer todayCommitsCount = getTodayCommitsCount(userProjects, userEmail);

        Integer todayMergeRequestsCount = getTodayMergeRequestsCount(userProjects, userEmail);

        return new UsersProjectsStats(totalProjectsCount, todayCommitsCount, todayMergeRequestsCount);
    }


    private Integer getTodayCommitsCount(List<UserProject> userProjects, String userEmail) {
        return userProjects.stream()
                .map(userProject -> gitLabServiceClient.fetchTodayCommitsCount(userProject.getProject().getId(),
                        userProject.getProject().getToken(), userEmail))
                .reduce(0, Integer::sum);
    }

    private Integer getTodayMergeRequestsCount(List<UserProject> userProjects, String userEmail) {
        return userProjects.stream()
                .map(userProject -> gitLabServiceClient.fetchTodayMergeRequestsCount(userProject.getProject().getId(),
                        userProject.getProject().getToken(), userEmail))
                .reduce(0, Integer::sum);
    }

    public ProjectStats getProjectStats(String token, Long projectId) {

        Long userId = getUserIdByToken(token);

        String projectAccessToken = projectRepository.findById(projectId)
                .orElseThrow(() -> new IllegalArgumentException("Project not found"))
                .getToken();

        Integer todayCommitsCount = gitLabServiceClient.fetchTodayCommitsCount(projectId, projectAccessToken,
                getUserEmailByToken(token));

        Integer totalMergeRequestsCount = getTotalMergeRequestsCount(getUserProjectsBy(userId));

        Integer todayMergeRequestsCount = gitLabServiceClient.fetchTodayMergeRequestsCount(projectId,
                projectAccessToken,
                getUserEmailByToken(token));

        return new ProjectStats(todayCommitsCount, totalMergeRequestsCount, todayMergeRequestsCount);
    }

    private Integer getTotalMergeRequestsCount(List<UserProject> userProjects) {
        return userProjects.stream()
                .map(userProject -> gitLabServiceClient.fetchMergeRequestsCount(userProject.getProject().getId(),
                        userProject.getProject().getToken()))
                .reduce(0, Integer::sum);
    }

    public ProjectDashboardDto getProjectDashboard(String token, Long projectId) {

        // TODO: Implement this business logic
        DashboardDto dashboardDto = (DashboardDto) redisTemplate.opsForValue().get("dashboard:" + projectId);
        return ProjectDashboardDto.from(dashboardDto);
    }

}
